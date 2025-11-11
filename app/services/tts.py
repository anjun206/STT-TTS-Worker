# tts.py
from __future__ import annotations

import json
import logging
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Dict

import torch
import torchaudio
from pydub import AudioSegment

from config import get_job_paths
from services.transcript_store import (
    COMPACT_ARCHIVE_NAME,
    load_compact_transcript,
    segment_views,
)

logger = logging.getLogger(__name__)


def _inject_cosyvoice_paths() -> None:
    """Ensure CosyVoice's source tree (and Matcha-TTS) are importable."""
    cosy_root = Path(os.getenv("COSYVOICE_DIR", "/opt/CosyVoice"))
    candidates = (
        cosy_root,
        cosy_root / "third_party" / "Matcha-TTS",
    )
    for path in candidates:
        try:
            if path.is_dir():
                path_str = str(path)
                if path_str not in sys.path:
                    sys.path.append(path_str)
        except Exception:
            continue


_inject_cosyvoice_paths()
try:
    from cosyvoice.cli.cosyvoice import CosyVoice2  # type: ignore
    from cosyvoice.utils.file_utils import load_wav  # type: ignore

    COSYVOICE_AVAILABLE = True
except Exception as exc:  # noqa: F841
    CosyVoice2 = None  # type: ignore
    load_wav = None  # type: ignore
    COSYVOICE_AVAILABLE = False


PROMPT_STT_MODEL_ID = os.getenv("COSYVOICE_PROMPT_STT_MODEL", "large-v3")
DEFAULT_TTS_DEVICE = (
    os.getenv("TTS_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
).lower()
PROMPT_STT_DEVICE = (
    os.getenv("COSYVOICE_PROMPT_STT_DEVICE") or DEFAULT_TTS_DEVICE
).lower()
PROMPT_STT_COMPUTE = os.getenv("COSYVOICE_PROMPT_STT_COMPUTE")

_PROMPT_CACHE: Dict[str, str] = {}


def _resolve_cosyvoice_model_dir() -> Path:
    """Pick the CosyVoice2 model directory from env or default location."""
    raw = (
        os.getenv("COSYVOICE2_MODEL_DIR") or os.getenv("COSYVOICE_MODEL_DIR") or ""
    ).strip()
    if raw:
        return Path(raw).expanduser()
    cosy_root = Path(os.getenv("COSYVOICE_DIR", "/opt/CosyVoice"))
    return cosy_root / "pretrained_models" / "CosyVoice2-0.5B"


@lru_cache(maxsize=1)
def _get_cosyvoice2():
    """Lazily load CosyVoice2 (if available)."""
    if not COSYVOICE_AVAILABLE or CosyVoice2 is None or load_wav is None:
        raise RuntimeError("CosyVoice2 backend is not installed/configured.")
    model_dir = _resolve_cosyvoice_model_dir()
    if not model_dir.is_dir():
        raise FileNotFoundError(f"CosyVoice2 model directory not found: {model_dir}")
    fp16 = DEFAULT_TTS_DEVICE == "cuda" and torch.cuda.is_available()
    cv = CosyVoice2(
        str(model_dir),
        load_jit=False,
        load_trt=False,
        load_vllm=False,
        fp16=fp16,
    )
    return cv, load_wav


@lru_cache(maxsize=1)
def _get_prompt_stt_model():
    """Load the fast-whisper model (large-v3 by default) for prompt extraction."""
    from faster_whisper import WhisperModel

    device = PROMPT_STT_DEVICE if torch.cuda.is_available() else "cpu"
    if device not in {"cuda", "cpu"}:
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    compute_type = PROMPT_STT_COMPUTE
    if not compute_type:
        compute_type = "float16" if device == "cuda" else "int8"
    return WhisperModel(
        PROMPT_STT_MODEL_ID,
        device=device,
        compute_type=compute_type,
    )


def _transcribe_prompt_text(sample_path: Path) -> str:
    """ASR the reference sample to obtain a light-weight prompt text."""
    cache_key = str(sample_path.resolve())
    if cache_key in _PROMPT_CACHE:
        return _PROMPT_CACHE[cache_key]

    text = ""
    try:
        model = _get_prompt_stt_model()
        segments, _ = model.transcribe(
            str(sample_path),
            beam_size=1,
            best_of=1,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 200},
        )
        collected = []
        for seg in segments:
            piece = getattr(seg, "text", "").strip()
            if piece:
                collected.append(piece)
        text = " ".join(collected).strip()
    except Exception as exc:
        logger.warning("Prompt STT fallback failed for %s: %s", sample_path, exc)
    _PROMPT_CACHE[cache_key] = text
    return text


def _prepare_self_reference_samples(
    vocals_audio: AudioSegment, segments: list[dict], out_dir: Path
) -> Dict[str, Path]:
    """Extract one reference sample per speaker for self-referencing."""
    references: Dict[str, Path] = {}
    out_dir.mkdir(parents=True, exist_ok=True)
    for seg in segments:
        speaker = seg.get("speaker")
        if not speaker or speaker in references:
            continue
        start = max(0.0, float(seg.get("start", 0.0)))
        end = float(seg.get("end", start))
        if end <= start:
            end = start + 1.0
        start_ms = int(start * 1000)
        end_ms = int(end * 1000)
        if end_ms - start_ms < 400:
            end_ms = start_ms + 400
        sample_audio = vocals_audio[start_ms:end_ms]
        if len(sample_audio) == 0:
            continue
        ref_path = out_dir / f"{speaker}_self_ref.wav"
        sample_audio.set_frame_rate(16000).set_channels(1).export(
            ref_path, format="wav"
        )
        references[speaker] = ref_path
    return references


def _resolve_path(candidate: str, paths) -> Path:
    """Resolve a user-provided path relative to the job directories."""
    path_obj = Path(candidate).expanduser()
    if path_obj.is_absolute():
        return path_obj
    search_roots = (
        paths.interim_dir,
        paths.outputs_dir,
        paths.input_dir,
        Path("."),
    )
    for root in search_roots:
        resolved = (root / path_obj).resolve()
        if resolved.exists():
            return resolved
    return (paths.interim_dir / path_obj).resolve()


def _select_voice_sample(
    seg: dict,
    speaker_refs: Dict[str, Path],
    paths,
    global_sample: Path | None = None,
) -> Path:
    """Pick the best voice-sample path for the segment."""
    override = seg.get("voice_sample_path") or seg.get("voice_sample")
    if isinstance(override, str) and override.strip():
        resolved = _resolve_path(override.strip(), paths)
        if not resolved.is_file():
            raise FileNotFoundError(f"Voice sample override not found: {resolved}")
        return resolved
    if global_sample and global_sample.is_file():
        return global_sample
    speaker = seg.get("speaker")
    if not speaker:
        raise ValueError("Segment is missing speaker information.")
    ref = speaker_refs.get(speaker)
    if not ref or not ref.is_file():
        raise FileNotFoundError(
            f"No self-reference audio prepared for speaker {speaker}."
        )
    return ref


def _resolve_prompt_text(
    seg: dict, sample_path: Path, prompt_text_override: str | None = None
) -> str:
    """Return prompt text, generating it via ASR if missing."""
    if prompt_text_override and prompt_text_override.strip():
        return prompt_text_override.strip()
    prompt = (
        seg.get("prompt_text") or seg.get("prompt") or seg.get("reference_text") or ""
    )
    prompt = prompt.strip()
    if prompt:
        return prompt

    prompt = _transcribe_prompt_text(sample_path)
    if prompt:
        return prompt
    return (seg.get("text") or "").strip()


def _synthesize_with_cosyvoice2(
    text: str, prompt_text: str, sample_path: Path, output_path: Path
) -> None:
    """Run CosyVoice2 zero-shot inference; fail loudly if cloning is impossible."""
    cv, load_wav_fn = _get_cosyvoice2()
    prompt_speech_16k = load_wav_fn(str(sample_path), 16000)
    generator = cv.inference_zero_shot(
        text,
        prompt_text,
        prompt_speech_16k,
        stream=False,
        text_frontend=False,
    )
    chunks = [item["tts_speech"] for item in generator]
    if not chunks:
        raise RuntimeError("CosyVoice2 zero-shot 클로닝 결과가 비어 있습니다.")

    waveform = chunks[0] if len(chunks) == 1 else torch.cat(chunks, dim=-1)
    sample_rate = getattr(cv, "sample_rate", 24000)
    torchaudio.save(str(output_path), waveform, sample_rate)


def generate_tts(
    job_id: str,
    target_lang: str,
    voice_sample_path: Path | None = None,
    prompt_text_override: str | None = None,
):
    """Use CosyVoice2 (if available) to synthesize translated segments."""
    if not COSYVOICE_AVAILABLE:
        raise RuntimeError(
            "CosyVoice2 백엔드가 설치되지 않아 보이스 클로닝을 진행할 수 없습니다."
        )
    paths = get_job_paths(job_id)
    trans_path = paths.trg_sentence_dir / "translated.json"
    if not trans_path.is_file():
        raise FileNotFoundError(
            "Translated text not found. Run translation stage first."
        )
    with open(trans_path, "r", encoding="utf-8") as f:
        translation_entries = json.load(f)
    translation_map: Dict[int, dict] = {}
    for entry in translation_entries:
        seg_idx = entry.get("seg_idx")
        try:
            idx = int(seg_idx)
        except (TypeError, ValueError):
            continue
        translation_map[idx] = entry

    transcript_archive = paths.src_sentence_dir / COMPACT_ARCHIVE_NAME
    bundle = load_compact_transcript(transcript_archive)
    base_segments = segment_views(bundle)
    if not base_segments:
        raise RuntimeError("No source segments found. Run ASR stage first.")

    vocals_path = paths.vid_speaks_dir / "vocals.wav"
    if not vocals_path.is_file():
        raise FileNotFoundError("Vocals track not found. Run Demucs stage first.")
    vocals_audio = AudioSegment.from_wav(str(vocals_path))

    tts_dir = paths.vid_tts_dir
    tts_dir.mkdir(parents=True, exist_ok=True)
    ref_segments = [
        {
            "speaker": seg.speaker,
            "start": seg.start_seconds,
            "end": seg.end_seconds,
        }
        for seg in base_segments
    ]
    speaker_refs = _prepare_self_reference_samples(vocals_audio, ref_segments, tts_dir)

    normalized_user_sample = None
    if voice_sample_path:
        candidate = Path(voice_sample_path)
        if candidate.is_file():
            try:
                user_audio = AudioSegment.from_file(str(candidate))
                normalized_user_sample = tts_dir / "user_voice_sample.wav"
                user_audio.set_frame_rate(16000).set_channels(1).export(
                    normalized_user_sample, format="wav"
                )
            except Exception as exc:
                logger.warning(
                    "사용자 보이스 샘플 변환 실패 (%s), 원본을 그대로 사용합니다: %s",
                    candidate,
                    exc,
                )
                normalized_user_sample = candidate
        else:
            logger.warning("제공된 보이스 샘플을 찾을 수 없습니다: %s", candidate)

    synthesized_segments = []
    for seg in base_segments:
        override = translation_map.get(seg.idx, {})
        text = override.get("translation") or seg.text
        speaker = override.get("speaker") or seg.speaker
        segment_start = seg.start_seconds
        segment_end = seg.end_seconds
        duration = max(0.0, seg.duration_seconds)
        output_file = tts_dir / f"{speaker}_{segment_start:.2f}.wav"

        seg_payload = {
            "speaker": speaker,
            "text": text,
            "start": segment_start,
            "end": segment_end,
            "voice_sample_path": override.get("voice_sample_path")
            or override.get("voice_sample"),
            "voice_sample": override.get("voice_sample"),
            "prompt_text": override.get("prompt_text") or override.get("prompt"),
            "reference_text": override.get("reference_text"),
        }
        effective_sample = _select_voice_sample(
            seg_payload, speaker_refs, paths, normalized_user_sample
        )
        prompt_text = _resolve_prompt_text(
            seg_payload, effective_sample, prompt_text_override
        )
        _synthesize_with_cosyvoice2(
            text=text,
            prompt_text=prompt_text,
            sample_path=effective_sample,
            output_path=output_file,
        )

        synthesized_segments.append(
            {
                "segment_id": seg.segment_id(),
                "seg_idx": seg.idx,
                "speaker": speaker,
                "start": segment_start,
                "end": segment_end,
                "target_duration": duration,
                "audio_file": str(output_file),
                "voice_sample": str(effective_sample),
                "prompt_text": prompt_text,
                "tts_backend": "cosyvoice2",
            }
        )
    meta_path = tts_dir / "segments.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(synthesized_segments, f, ensure_ascii=False, indent=2)
    return synthesized_segments
