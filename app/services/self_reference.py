# services/self_reference.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Sequence

from pydub import AudioSegment

from .transcript_store import SegmentView

logger = logging.getLogger(__name__)

# self-reference 프롬프트 오디오의 최대 길이 (15초)
MAX_REF_DURATION_MS = 15_000


@dataclass
class SpeakerReferenceSample:
    speaker: str
    audio_path: Path
    text: str
    segment_idx: int
    segment_id: str
    start_ms: int
    end_ms: int
    audio_duration_ms: int
    score: float | None = None

    @property
    def segment_duration_ms(self) -> int:
        return max(0, self.end_ms - self.start_ms)

    def to_payload(self, base_dir: Path) -> dict:
        try:
            audio_rel = self.audio_path.relative_to(base_dir)
            audio_value = str(audio_rel)
        except ValueError:
            audio_value = str(self.audio_path)
        return {
            "audio": audio_value,
            "text": self.text,
            "segment_idx": self.segment_idx,
            "segment_id": self.segment_id,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "segment_duration_ms": self.segment_duration_ms,
            "audio_duration_ms": self.audio_duration_ms,
            "score": self.score,
        }

    @classmethod
    def from_payload(
        cls, speaker: str, payload: Any, base_dir: Path
    ) -> "SpeakerReferenceSample":
        if isinstance(payload, str):
            audio_value = payload
            meta: dict[str, Any] = {}
        elif isinstance(payload, dict):
            audio_value = payload.get("audio") or payload.get("path") or ""
            meta = payload
        else:
            audio_value = ""
            meta = {}

        if not audio_value:
            audio_value = f"{speaker}_self_ref.wav"

        audio_path = Path(audio_value)
        if not audio_path.is_absolute():
            audio_path = (base_dir / audio_path).resolve()

        def _int_value(key: str, default: int = 0) -> int:
            value = meta.get(key)
            try:
                return int(value)
            except (TypeError, ValueError):
                return default

        segment_idx = _int_value("segment_idx", -1)
        start_ms = _int_value("start_ms", 0)
        end_ms = _int_value("end_ms", start_ms)
        seg_duration = _int_value("segment_duration_ms", max(0, end_ms - start_ms))
        if end_ms <= start_ms:
            end_ms = start_ms + seg_duration
        audio_duration = _int_value("audio_duration_ms", seg_duration)

        score_val = meta.get("score")
        try:
            score = float(score_val) if score_val is not None else None
        except (TypeError, ValueError):
            score = None

        text = (meta.get("text") or "").strip()
        segment_id = meta.get("segment_id")
        if not segment_id:
            segment_suffix = f"{segment_idx:04d}" if segment_idx >= 0 else "unknown"
            segment_id = f"segment_{segment_suffix}"

        return cls(
            speaker=speaker,
            audio_path=audio_path,
            text=text,
            segment_idx=segment_idx,
            segment_id=segment_id,
            start_ms=start_ms,
            end_ms=end_ms,
            audio_duration_ms=audio_duration,
            score=score,
        )


def serialize_reference_mapping(
    references: Dict[str, SpeakerReferenceSample], base_dir: Path
) -> Dict[str, Any]:
    return {speaker: ref.to_payload(base_dir) for speaker, ref in references.items()}


def deserialize_reference_mapping(
    payload: Dict[str, Any], base_dir: Path
) -> Dict[str, SpeakerReferenceSample]:
    mapping: Dict[str, SpeakerReferenceSample] = {}
    for speaker, entry in payload.items():
        try:
            mapping[speaker] = SpeakerReferenceSample.from_payload(
                speaker, entry, base_dir
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to parse speaker reference for %s: %s", speaker, exc)
    return mapping


def prepare_self_reference_samples(
    vocals_audio: AudioSegment, segments: Sequence[SegmentView], out_dir: Path
) -> Dict[str, SpeakerReferenceSample]:
    """
    화자별로 STT 세그먼트 중에서:

    1) 길이 15초 이하인 세그먼트들 중 점수가 가장 높은 것 우선 선택.
    2) 만약 15초 이하 세그먼트가 하나도 없다면, 길이와 상관 없이
       점수가 가장 높은 세그먼트를 선택한 뒤, 오디오를 15초로 잘라 사용.

    반환: speaker -> SpeakerReferenceSample
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # speaker 별로:
    # - best_short: duration <= 15초 중 최고 점수
    # - best_any  : 전체 세그먼트 중 최고 점수 (fallback용)
    best_short: dict[str, tuple[SegmentView, float]] = {}
    best_any: dict[str, tuple[SegmentView, float]] = {}

    for seg in segments:
        speaker = getattr(seg, "speaker", None)
        text = (getattr(seg, "text", "") or "").strip()
        if not speaker or not text:
            continue

        duration_ms = getattr(seg, "duration_ms", 0) or 0
        if duration_ms <= 0:
            continue

        score_val = getattr(seg, "score", None)
        try:
            score = float(score_val) if score_val is not None else 0.0
        except (TypeError, ValueError):
            score = 0.0

        # 전체 최고 세그먼트 갱신
        cur_any = best_any.get(speaker)
        if cur_any is None or score > cur_any[1]:
            best_any[speaker] = (seg, score)

        # 15초 이하 세그먼트 중 최고 세그먼트 갱신
        if duration_ms <= MAX_REF_DURATION_MS:
            cur_short = best_short.get(speaker)
            if cur_short is None or score > cur_short[1]:
                best_short[speaker] = (seg, score)

    references: Dict[str, SpeakerReferenceSample] = {}
    total_length = len(vocals_audio)

    # speaker 별로 최종 세그먼트 선택
    for speaker, (fallback_seg, fallback_score) in best_any.items():
        short_entry = best_short.get(speaker)
        if short_entry is not None:
            seg, score = short_entry
        else:
            seg, score = fallback_seg, fallback_score

        # 세그먼트 구간 → 오디오 인덱스
        start_ms = max(0, min(seg.start_ms, total_length))
        end_ms = min(seg.end_ms, total_length)
        if end_ms <= start_ms:
            continue

        # 오디오 길이를 최대 15초로 제한
        if end_ms - start_ms > MAX_REF_DURATION_MS:
            end_ms = min(start_ms + MAX_REF_DURATION_MS, total_length)

        sample_audio = vocals_audio[start_ms:end_ms]
        if len(sample_audio) == 0:
            continue

        final_audio = sample_audio.set_frame_rate(16000).set_channels(1)
        if len(final_audio) == 0:
            continue

        ref_path = out_dir / f"{speaker}_self_ref.wav"
        final_audio.export(ref_path, format="wav")

        references[speaker] = SpeakerReferenceSample(
            speaker=speaker,
            audio_path=ref_path,
            text=seg.text.strip(),
            segment_idx=seg.idx,
            segment_id=seg.segment_id(),
            start_ms=seg.start_ms,
            end_ms=seg.end_ms,
            audio_duration_ms=len(final_audio),
            score=seg.score,
        )

    logger.info(
        "Prepared %d self-reference samples (max %d ms each)",
        len(references),
        MAX_REF_DURATION_MS,
    )
    return references
