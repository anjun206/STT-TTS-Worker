# app/text_fit.py
import re

def estimate_char_budget(duration_sec: float, lang: str) -> int:
    # 대략적인 발화 속도 휴리스틱 (필요시 조정)
    # en: 13~15 chars/s, ja/ko: 7~9 chars/s 근사
    speed = {"en": 14.0, "ja": 8.0, "ko": 8.0}.get(lang, 12.0)
    return max(1, int(duration_sec * speed))

def simple_fit(text: str, budget: int) -> str:
    t = re.sub(r"\s+", " ", text).strip()
    if len(t) <= budget:
        return t
    # 군더더기 제거 위주 압축
    t = t.replace(", ", " ")
    t = re.sub(r"\b(정말|아주|매우|진짜|Please|kindly|just)\b\s*", "", t, flags=re.I)
    t = re.sub(r"\s+", " ", t).strip()
    if len(t) > budget:
        t = t[:budget].rstrip()
        t = re.sub(r"[\s,;:.!?\-]+$", "", t)
    return t
