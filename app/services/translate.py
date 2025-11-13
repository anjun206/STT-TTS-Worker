"""Vertex AI Gemini(Flash/Flash-Lite) 기반 기계번역 서비스.

변경 요약
- googletrans 제거, Vertex AI Gemini로 대체.
- 최소 10개 세그먼트 단위 배치 호출.
- JSON 스키마(response_schema)로 '입력 N개 → 출력 N개' 강제.
- 각 항목 길이 유사(기본 0.9~1.1배) 검증 및 위반 항목만 1회 자동 보정.
- ENV: VERTEX_PROJECT_ID, VERTEX_LOCATION(기본 asia-northeast3), VERTEX_GEMINI_MODEL,
       GOOGLE_APPLICATION_CREDENTIALS 또는 VERTEX_SA_PATH, MT_MIN_BATCH_SIZE(기본 10),
       MT_LEN_RATIO_LOW/HIGH(기본 0.9/1.1)
"""

from __future__ import annotations

import json
import os
import shutil
from typing import Iterable, List, Dict, Any, Tuple

from configs import get_job_paths
from services.transcript_store import (
    COMPACT_ARCHIVE_NAME,
    load_compact_transcript,
    segment_views,
)

# 선택 의존성: 라이브러리가 없어도 모듈 import 자체는 되도록 지연 임포트
_VERTEX_AVAILABLE = True
try:
    import vertexai  # type: ignore
    from vertexai.generative_models import (  # type: ignore
        GenerativeModel,
        GenerationConfig,
        Schema,
        Type,
    )
    from google.oauth2 import service_account  # type: ignore
except Exception:  # pragma: no cover
    _VERTEX_AVAILABLE = False


def _env_str(key: str, default: str | None = None) -> str | None:
    v = os.getenv(key)
    return v if v is not None and v != "" else default


def _chunked(seq: Iterable[Any], size: int) -> Iterable[List[Any]]:
    """이터러블을 최대 `size` 길이의 리스트들로 잘라 순차 반환."""
    batch: List[Any] = []
    for item in seq:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def _length_bounds(srcs: List[str], low: float, high: float) -> List[Tuple[int, int]]:
    bounds: List[Tuple[int, int]] = []
    for s in srcs:
        L = max(1, len(s))
        lo = max(1, int(L * low))
        hi = max(lo, int(L * high))
        bounds.append((lo, hi))
    return bounds


class GeminiTranslator:
    """Vertex AI Gemini 기반 번역기.

    - ENV로 프로젝트/리전/모델/인증을 읽어 초기화.
    - JSON 전용 응답 + 스키마 강제(response_schema)로 파싱 안정성 확보.
    """

    def __init__(self) -> None:
        if not _VERTEX_AVAILABLE:
            raise RuntimeError(
                "Vertex AI libraries not installed. Add google-cloud-aiplatform to requirements."
            )

        # 모델/리전 기본값: 서울 리전, Flash-Lite(또는 Flash로 변경)
        self.model_name = _env_str("VERTEX_GEMINI_MODEL", "gemini-2.5-flash-lite")
        self.location = _env_str("VERTEX_LOCATION", "asia-northeast3")
        self.project_id = _env_str("VERTEX_PROJECT_ID")

        # 서비스 계정 JSON
        sa_path = _env_str("VERTEX_SA_PATH") or _env_str(
            "GOOGLE_APPLICATION_CREDENTIALS"
        )

        creds = None
        if sa_path and os.path.isfile(sa_path):
            # 프로젝트 ID가 없으면 JSON에서 복구 시도
            if not self.project_id:
                try:
                    with open(sa_path, "r", encoding="utf-8") as f:
                        j = json.load(f)
                        self.project_id = j.get("project_id")
                except Exception:
                    pass
            try:
                creds = service_account.Credentials.from_service_account_file(
                    sa_path, scopes=["https://www.googleapis.com/auth/cloud-platform"]
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load service account JSON at {sa_path}: {e}"
                )

        if not self.project_id:
            raise RuntimeError(
                "VERTEX_PROJECT_ID is required (or set in service account JSON)."
            )

        vertexai.init(
            project=self.project_id, location=self.location, credentials=creds
        )
        self._model = GenerativeModel(self.model_name)

    def translate_batch(
        self,
        items: List[Dict[str, Any]],
        target_lang: str,
        src_lang: str | None = None,
        ratio_low: float | None = None,
        ratio_high: float | None = None,
    ) -> List[Dict[str, Any]]:
        """배치 번역 수행.

        items: [{"seg_idx": int, "text": str}, ...]
        반환: [{"seg_idx": int, "translation": str}, ...] (입력 개수와 동일)
        """
        n = len(items)
        if n == 0:
            return []

        # 길이 제약 비율
        r_low = (
            float(_env_str("MT_LEN_RATIO_LOW", "0.9") or "0.9")
            if ratio_low is None
            else ratio_low
        )
        r_high = (
            float(_env_str("MT_LEN_RATIO_HIGH", "1.1") or "1.1")
            if ratio_high is None
            else ratio_high
        )

        # 입력 준비
        src_texts = [str(o["text"]) for o in items]
        seg_idxs = [int(o["seg_idx"]) for o in items]
        bounds = _length_bounds(src_texts, r_low, r_high)

        # 스키마: 길이 N의 배열, 각 항목에 seg_idx/translation/char_count 강제
        item_schema = Schema(
            type=Type.OBJECT,
            properties={
                "seg_idx": Schema(type=Type.INTEGER),
                "translation": Schema(type=Type.STRING),
                "char_count": Schema(type=Type.INTEGER),
            },
            required=["seg_idx", "translation", "char_count"],
        )
        array_schema = Schema(
            type=Type.ARRAY,
            items=item_schema,
            min_items=n,
            max_items=n,
        )

        sys = (
            "You are a professional subtitle translator.\n"
            "- Return EXACTLY one JSON array of length N in the SAME ORDER as input.\n"
            "- Do NOT merge or split sentences. No extra notes, no numbering.\n"
            "- Each item must have: seg_idx(int), translation(string), char_count(int=length of translation in characters).\n"
            "- Keep punctuation natural; preserve names, numbers, units."
        )
        rules = "\n".join(
            f"[{i}] target char range: {lo}..{hi}" for i, (lo, hi) in enumerate(bounds)
        )
        src_lang_txt = (
            f"Source language: {src_lang}"
            if src_lang
            else "Source language: auto-detect"
        )
        user = (
            f"N={n}\n"
            f"{src_lang_txt}\nTarget language: {target_lang}\n\n"
            f"Length constraints per index:\n{rules}\n\n"
            "Inputs (keep order and seg_idx):\n"
            + "\n".join(
                f"[{i}] seg_idx={seg_idxs[i]} text={src_texts[i]}" for i in range(n)
            )
            + "\n\nReturn JSON ONLY per schema."
        )

        gen_cfg = GenerationConfig(
            temperature=0.1,
            max_output_tokens=4096,
            response_mime_type="application/json",
            response_schema=array_schema,
        )

        # 1차 생성
        resp = self._model.generate_content(
            contents=[sys, user],
            generation_config=gen_cfg,
        )
        items_out = self._parse_json_array(self._extract_text(resp))

        # 길이/키 검증 & 보정 대상 수집
        fixed = self._normalize_and_validate(items_out, seg_idxs, bounds)

        # 어긋난 인덱스가 있으면 1회 보정 요청
        bad_idxs = fixed["bad_idxs"]
        if bad_idxs:
            prev_json = json.dumps(fixed["norm"], ensure_ascii=False)
            fix_instr = (
                "You previously produced translations with length violations.\n"
                "Rewrite ONLY the listed items to fit the specified char ranges.\n"
                "Return the FULL JSON array of length N. Non-listed items MUST be copied AS-IS.\n\n"
                "Targets:\n"
                + "\n".join(
                    f"[{i}] seg_idx={seg_idxs[i]} range={bounds[i][0]}..{bounds[i][1]} "
                    f"current='{fixed['norm'][i]['translation']}'"
                    for i in bad_idxs
                )
            )
            resp2 = self._model.generate_content(
                contents=[sys, user, "Previous JSON output:\n" + prev_json, fix_instr],
                generation_config=gen_cfg,
            )
            items_out2 = self._parse_json_array(self._extract_text(resp2))
            fixed = self._normalize_and_validate(items_out2, seg_idxs, bounds)

        # 최종 translations 반환(누락/실패 시 원문 대체)
        final_objs: List[Dict[str, Any]] = []
        for i in range(n):
            final_objs.append(
                {
                    "seg_idx": seg_idxs[i],
                    "translation": (
                        fixed["norm"][i]["translation"]
                        if fixed["norm"][i]["translation"] is not None
                        else src_texts[i]
                    ),
                }
            )
        return final_objs

    @staticmethod
    def _extract_text(resp: Any) -> str:
        # SDK 버전에 따라 텍스트 접근 경로가 다를 수 있음
        # 우선 문자열 변환 시도
        try:
            # aiplatform 최신 SDK는 resp.candidates[0].content.parts[0].text 형태가 일반적
            cands = getattr(resp, "candidates", None)
            if cands:
                content = cands[0].content
                parts = getattr(content, "parts", None)
                if parts and getattr(parts[0], "text", None):
                    return parts[0].text
        except Exception:
            pass
        try:
            if hasattr(resp, "text") and isinstance(resp.text, str):
                return resp.text
        except Exception:
            pass
        return str(resp)

    @staticmethod
    def _parse_json_array(text: str) -> List[Dict[str, Any]]:
        # 1) 순수 JSON 시도
        try:
            obj = json.loads(text)
            if isinstance(obj, list):
                return obj  # type: ignore[return-value]
        except Exception:
            pass
        # 2) 주변 텍스트에서 JSON 배열 부분 추출 시도
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            snippet = text[start : end + 1]
            try:
                obj = json.loads(snippet)
                if isinstance(obj, list):
                    return obj  # type: ignore[return-value]
            except Exception:
                pass
        # 3) 실패 시 빈 배열
        return []

    @staticmethod
    def _normalize_and_validate(
        parsed: List[Dict[str, Any]],
        seg_idxs_in_order: List[int],
        bounds: List[Tuple[int, int]],
    ) -> Dict[str, Any]:
        """모델 출력 배열을 입력 순서로 정렬/정규화 + 길이검증."""
        n = len(seg_idxs_in_order)
        # seg_idx -> obj 매핑
        by_idx: Dict[int, Dict[str, Any]] = {}
        for obj in parsed or []:
            try:
                si = int(obj.get("seg_idx"))
                tr = obj.get("translation")
                cc = obj.get("char_count")
                by_idx[si] = {
                    "translation": tr if isinstance(tr, str) else None,
                    "char_count": int(cc) if isinstance(cc, int) else None,
                }
            except Exception:
                continue

        norm: List[Dict[str, Any]] = []
        bad_idxs: List[int] = []
        for i, seg_idx in enumerate(seg_idxs_in_order):
            cur = by_idx.get(seg_idx, {"translation": None, "char_count": None})
            norm.append(cur)
            # 길이 검증
            if cur["translation"] is None or cur["char_count"] is None:
                bad_idxs.append(i)
            else:
                lo, hi = bounds[i]
                cc = int(cur["char_count"])
                if not (lo <= cc <= hi):
                    bad_idxs.append(i)
        return {"norm": norm, "bad_idxs": bad_idxs}


def _merge_batches(
    original_items: List[Dict[str, Any]], batch_outputs: List[List[Dict[str, Any]]]
) -> List[Dict[str, Any]]:
    """배치 결과를 원본 순서로 병합.

    누락된 번역은 원문 텍스트로 대체.
    """
    merged: Dict[int, str] = {}
    for out in batch_outputs:
        for obj in out:
            if isinstance(obj.get("seg_idx"), int) and isinstance(
                obj.get("translation"), str
            ):
                merged[obj["seg_idx"]] = obj["translation"]

    result: List[Dict[str, Any]] = []
    for item in original_items:
        idx = int(item["seg_idx"])  # type: ignore[arg-type]
        txt = str(item["text"])  # type: ignore[arg-type]
        result.append({"seg_idx": idx, "translation": merged.get(idx, txt)})
    return result


def translate_transcript(job_id: str, target_lang: str):
    """전사된 구간 텍스트를 지정 언어로 번역.

    - 기본적으로 Vertex AI Gemini Flash-Lite 사용(ENV로 변경 가능)
    - 세그먼트를 최소 10개 단위로 묶어 배치 호출
    - '입력 N개 → 출력 N개' 강제 및 길이 유사 검증/보정
    - 결과를 translated.json으로 저장하고 outputs에도 복사
    """
    paths = get_job_paths(job_id)
    transcript_path = paths.src_sentence_dir / COMPACT_ARCHIVE_NAME
    if not transcript_path.is_file():
        raise FileNotFoundError("Transcript not found. Run ASR stage first.")

    bundle = load_compact_transcript(transcript_path)
    seg_views = segment_views(bundle)

    # MT 입력 준비
    items = [{"seg_idx": s.idx, "text": s.text} for s in seg_views]

    min_batch = int(_env_str("MT_MIN_BATCH_SIZE", "10") or "10")
    if min_batch < 1:
        min_batch = 10

    translator = GeminiTranslator()

    batch_outputs: List[List[Dict[str, Any]]] = []
    for batch in _chunked(items, size=min_batch):
        out = translator.translate_batch(batch, target_lang)
        batch_outputs.append(out)

    translated_segments = _merge_batches(items, batch_outputs)

    # 결과 저장
    trans_out_path = paths.trg_sentence_dir / "translated.json"
    trans_out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(trans_out_path, "w", encoding="utf-8") as f:
        json.dump(translated_segments, f, ensure_ascii=False, indent=2)
    paths.outputs_text_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(trans_out_path, paths.outputs_text_dir / "trg_translated.json")
    return translated_segments
