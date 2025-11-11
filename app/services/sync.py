# sync.py
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List

from pydub import AudioSegment

from config import get_job_paths
from services.transcript_store import (
    COMPACT_ARCHIVE_NAME,
    load_compact_transcript,
    segment_views,
)

MAX_SLOW_RATIO = float(os.getenv("SYNC_MAX_SLOW_RATIO", "1.1"))
LENGTH_TOLERANCE_MS = 20  # 밀리초


def _resolve_audio_path(path_str: str, fallback_dir: Path) -> Path:
    path = Path(path_str)
    if path.is_file():
        return path
    candidate = fallback_dir / path.name
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"TTS 오디오 파일을 찾을 수 없습니다: {path_str}")


def _time_stretch(input_path: Path, ratio: float) -> AudioSegment:
    """ratio>1이면 길이를 늘리고, ratio<1이면 줄임."""
    if ratio <= 0:
        raise ValueError("시간 비율(ratio)은 0보다 커야 합니다.")
    if abs(ratio - 1.0) < 0.01:
        return AudioSegment.from_file(str(input_path))
    tempo_factor = 1.0 / ratio
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        temp_output = Path(tmp.name)
    try:
        cmd = [
            "sox",
            str(input_path),
            str(temp_output),
            "tempo",
            f"{tempo_factor:.6f}",
        ]
        subprocess.run(cmd, check=True, timeout=600)  # 10분 타임아웃
        return AudioSegment.from_wav(str(temp_output))
    finally:
        if temp_output.exists():
            temp_output.unlink()


def _apply_balanced_padding(audio: AudioSegment, padding_ms: int) -> AudioSegment:
    if padding_ms <= 0:
        return audio
    lead = AudioSegment.silent(duration=padding_ms // 2)
    tail = AudioSegment.silent(duration=padding_ms - len(lead))
    return lead + audio + tail


def _sync_single_segment(
    audio_path: Path,
    target_ms: int,
    allow_ratio: float,
) -> tuple[AudioSegment, float, int, int]:
    audio = AudioSegment.from_file(str(audio_path))
    current_ms = len(audio)
    if current_ms <= 0:
        raise RuntimeError(f"빈 오디오 파일입니다: {audio_path}")

    ratio = target_ms / current_ms
    if ratio <= 0:
        raise RuntimeError("목표 길이가 잘못되었습니다.")

    ratio_to_apply = ratio
    padding_added = 0
    if ratio > 1.0:
        ratio_to_apply = min(ratio, allow_ratio)
    stretched = _time_stretch(audio_path, ratio_to_apply)

    if ratio > allow_ratio and len(stretched) < target_ms:
        padding_needed = target_ms - len(stretched)
        stretched = _apply_balanced_padding(stretched, padding_needed)
        padding_added = padding_needed

    if len(stretched) > target_ms + LENGTH_TOLERANCE_MS:
        stretched = stretched[:target_ms]
    elif len(stretched) < target_ms - LENGTH_TOLERANCE_MS:
        pad_ms = target_ms - len(stretched)
        stretched += AudioSegment.silent(duration=pad_ms)
        padding_added += pad_ms

    return stretched, ratio_to_apply, padding_added, current_ms


def sync_segments(job_id: str) -> List[Dict]:
    """
    번역/TTS 후 구간별 오디오를 원본 화자 길이에 맞게 보정합니다.
    - 길이가 더 길다면 배속 제한 없이 줄임
    - 길이가 짧다면 최대 1.1배까지 실제 오디오를 스트레치하고 남는 길이는 무음으로 보충
    """
    paths = get_job_paths(job_id)
    transcript_path = paths.src_sentence_dir / COMPACT_ARCHIVE_NAME
    tts_meta_path = paths.vid_tts_dir / "segments.json"
    if not transcript_path.is_file():
        raise FileNotFoundError(
            f"원본 전사({COMPACT_ARCHIVE_NAME})를 찾을 수 없습니다."
        )
    if not tts_meta_path.is_file():
        raise FileNotFoundError("TTS 세그먼트 메타데이터가 없습니다. /tts 단계를 먼저 실행하세요.")

    bundle = load_compact_transcript(transcript_path)
    base_segments = segment_views(bundle)
    src_lookup = {seg.segment_id(): seg for seg in base_segments}

    with open(tts_meta_path, "r", encoding="utf-8") as f:
        tts_segments = json.load(f)

    synced_dir = paths.vid_tts_dir / "synced"
    synced_dir.mkdir(parents=True, exist_ok=True)

    synced_metadata: List[Dict] = []
    for entry in tts_segments:
        seg_id = entry.get("segment_id")
        if not seg_id or seg_id not in src_lookup:
            raise KeyError(f"segment_id {seg_id} 에 해당하는 원본 구간이 없습니다.")
        source_seg = src_lookup[seg_id]
        target_duration = float(
            entry.get("target_duration")
            or source_seg.duration_seconds
        )
        target_ms = max(1, int(target_duration * 1000))

        audio_path = _resolve_audio_path(entry["audio_file"], paths.vid_tts_dir)
        synced_audio, ratio_applied, padding_ms, original_ms = _sync_single_segment(
            audio_path,
            target_ms,
            MAX_SLOW_RATIO,
        )
        output_path = synced_dir / Path(audio_path).name
        synced_audio.export(str(output_path), format="wav")

        source_duration_sec = round(target_ms / 1000, 3)
        orig_duration_sec = round(original_ms / 1000, 3)
        synced_duration_sec = round(len(synced_audio) / 1000, 3)
        synced_metadata.append(
            {
                "segment_id": seg_id,
                "source_duration": source_duration_sec,
                "tts_duration": orig_duration_sec,
                "synced_duration": synced_duration_sec,
                "ratio_target": round(target_ms / original_ms, 4),
                "ratio_applied": round(ratio_applied, 4),
                "padding_ms": padding_ms,
                "audio_file": str(output_path),
            }
        )

    meta_path = synced_dir / "segments_synced.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(synced_metadata, f, ensure_ascii=False, indent=2)
    return synced_metadata
