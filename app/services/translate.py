# translate.py
import json
import shutil
from config import get_job_paths
from services.transcript_store import (
    COMPACT_ARCHIVE_NAME,
    load_compact_transcript,
    segment_views,
)
from googletrans import Translator

translator = Translator()

def translate_transcript(job_id: str, target_lang: str):
    """전사된 구간 텍스트를 지정한 언어로 번역합니다."""
    paths = get_job_paths(job_id)
    transcript_path = paths.src_sentence_dir / COMPACT_ARCHIVE_NAME
    if not transcript_path.is_file():
        raise FileNotFoundError("Transcript not found. Run ASR stage first.")
    bundle = load_compact_transcript(transcript_path)
    segments = segment_views(bundle)

    translated_segments = []
    for seg in segments:
        text = seg.text
        # 번역 API/라이브러리를 사용해 텍스트 변환
        try:
            # 현재는 googletrans를 예제로 사용하며, 실서비스에서는 Gemini 등의 API로 대체
            result = translator.translate(text, dest=target_lang)
            translated_text = result.text
        except Exception as e:
            # 실패 시에는 원문을 그대로 사용하거나 별도 예외 처리를 수행
            translated_text = text
        translated_segments.append(
            {
                "seg_idx": seg.idx,
                "translation": translated_text,
            }
        )

    # 번역 결과 저장
    trans_out_path = paths.trg_sentence_dir / "translated.json"
    with open(trans_out_path, "w", encoding="utf-8") as f:
        json.dump(translated_segments, f, ensure_ascii=False, indent=2)
    shutil.copyfile(
        trans_out_path, paths.outputs_text_dir / "trg_translated.json"
    )
    return translated_segments
