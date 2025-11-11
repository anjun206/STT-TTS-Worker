from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

SCHEMA_VERSION = 1
COMPACT_ARCHIVE_NAME = "transcript.comp.json.gz"


def _to_ms(value) -> int | None:
    if value is None:
        return None
    try:
        return int(round(float(value) * 1000))
    except (TypeError, ValueError):
        return None


def _ms_to_seconds(value: int | None) -> float | None:
    if value is None:
        return None
    return round(value / 1000, 3)


def _quantize_score(value) -> int:
    if value is None:
        return 0
    try:
        score = float(value)
    except (TypeError, ValueError):
        score = 0.0
    score = max(0.0, min(1.0, score))
    return int(round(score * 255))


def _normalize_speaker(raw) -> str:
    if raw is None:
        return "unknown_speaker"
    if isinstance(raw, int):
        return f"SPEAKER_{raw:02d}"
    value = str(raw).strip()
    return value or "unknown_speaker"


def _ensure_speaker_index(name: str, table: dict[str, int], ordered: list[str]) -> int:
    if name in table:
        return table[name]
    idx = len(ordered)
    table[name] = idx
    ordered.append(name)
    return idx


def _ensure_vocab_index(
    token: str, table: dict[str, int], ordered: list[str]
) -> int:
    if token in table:
        return table[token]
    idx = len(ordered)
    table[token] = idx
    ordered.append(token)
    return idx


@dataclass(frozen=True)
class SegmentView:
    idx: int
    start_ms: int
    end_ms: int
    speaker: str
    text: str
    gap_after_ms: int | None
    gap_after_vad_ms: int | None
    word_start: int
    word_count: int
    overlap: bool
    orig: int | None

    @property
    def duration_ms(self) -> int:
        return max(0, self.end_ms - self.start_ms)

    @property
    def start_seconds(self) -> float:
        return self.start_ms / 1000.0

    @property
    def end_seconds(self) -> float:
        return self.end_ms / 1000.0

    @property
    def duration_seconds(self) -> float:
        return self.duration_ms / 1000.0

    def segment_id(self) -> str:
        return f"segment_{self.idx:04d}"

    def to_public_dict(self) -> dict:
        return {
            "idx": self.idx,
            "segment_id": self.segment_id(),
            "speaker": self.speaker,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "start": _ms_to_seconds(self.start_ms),
            "end": _ms_to_seconds(self.end_ms),
            "duration_ms": self.duration_ms,
            "duration": _ms_to_seconds(self.duration_ms),
            "text": self.text,
            "gap_after_ms": self.gap_after_ms,
            "gap_after_vad_ms": self.gap_after_vad_ms,
            "gap_after": _ms_to_seconds(self.gap_after_ms),
            "gap_after_vad": _ms_to_seconds(self.gap_after_vad_ms),
            "word_count": self.word_count,
            "overlap": self.overlap,
            "orig_segment_id": self.orig,
        }


def build_compact_transcript(
    aligned_segments: Sequence[dict], language: str | None = None
) -> dict:
    """Convert WhisperX-aligned segments into the compact schema."""
    speakers: list[str] = []
    speaker_index: dict[str, int] = {}
    vocab: list[str] = []
    vocab_index: dict[str, int] = {}
    compact_segments: list[dict] = []
    compact_words: list[list[int]] = []
    prev_end_ms: int | None = None

    for idx, seg in enumerate(aligned_segments):
        start_ms = _to_ms(seg.get("start"))
        end_ms = _to_ms(seg.get("end"))
        if start_ms is None or end_ms is None:
            continue
        text = (seg.get("text") or "").strip()
        speaker_name = _normalize_speaker(seg.get("speaker"))
        speaker_idx = _ensure_speaker_index(speaker_name, speaker_index, speakers)

        w_start = len(compact_words)
        words = seg.get("words") or []
        for word in words:
            token = (word.get("word") or "").strip()
            if not token:
                continue
            w_abs_start = _to_ms(word.get("start"))
            w_abs_end = _to_ms(word.get("end"))
            if w_abs_start is None or w_abs_end is None:
                continue
            offset_start = max(0, w_abs_start - start_ms)
            offset_end = max(offset_start, w_abs_end - start_ms)
            vocab_idx = _ensure_vocab_index(token, vocab_index, vocab)
            score_q = _quantize_score(word.get("score"))
            compact_words.append(
                [idx, offset_start, offset_end, vocab_idx, score_q]
            )
        w_count = len(compact_words) - w_start

        next_start_ms = (
            _to_ms(aligned_segments[idx + 1].get("start"))
            if idx + 1 < len(aligned_segments)
            else None
        )
        gap_after = (
            next_start_ms - end_ms if next_start_ms is not None else None
        )
        gap_after_vad = (
            max(gap_after, 0) if gap_after is not None else None
        )
        overlap = bool(prev_end_ms is not None and start_ms < prev_end_ms)
        prev_end_ms = end_ms if prev_end_ms is None else max(prev_end_ms, end_ms)

        compact_segments.append(
            {
                "s": start_ms,
                "e": end_ms,
                "sp": speaker_idx,
                "txt": text,
                "gap": [gap_after, gap_after_vad],
                "w_off": [w_start, w_count],
                "o": seg.get("id", idx),
                "ov": overlap,
            }
        )

    return {
        "v": SCHEMA_VERSION,
        "unit": "ms",
        "lang": language,
        "speakers": speakers,
        "segments": compact_segments,
        "vocab": vocab,
        "words": compact_words,
    }


def save_compact_transcript(bundle: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(bundle, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    with gzip.open(path, "wb") as fh:
        fh.write(payload)


def load_compact_transcript(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Transcript archive not found: {path}")
    data: bytes
    if path.suffix == ".gz":
        with gzip.open(path, "rb") as fh:
            data = fh.read()
    else:
        data = path.read_bytes()
    return json.loads(data.decode("utf-8"))


def segment_views(bundle: dict) -> List[SegmentView]:
    speakers = bundle.get("speakers") or []
    views: list[SegmentView] = []
    for idx, seg in enumerate(bundle.get("segments") or []):
        start_ms = int(seg.get("s") or 0)
        end_ms = int(seg.get("e") or start_ms)
        sp_idx = seg.get("sp")
        speaker = (
            speakers[sp_idx]
            if isinstance(sp_idx, int) and 0 <= sp_idx < len(speakers)
            else "unknown_speaker"
        )
        gap = seg.get("gap") or [None, None]
        w_off = seg.get("w_off") or [0, 0]
        views.append(
            SegmentView(
                idx=idx,
                start_ms=start_ms,
                end_ms=end_ms,
                speaker=speaker,
                text=seg.get("txt") or "",
                gap_after_ms=gap[0],
                gap_after_vad_ms=gap[1],
                word_start=w_off[0],
                word_count=w_off[1],
                overlap=bool(seg.get("ov")),
                orig=seg.get("o"),
            )
        )
    return views


def segment_preview(bundle: dict) -> list[dict]:
    return [view.to_public_dict() for view in segment_views(bundle)]
