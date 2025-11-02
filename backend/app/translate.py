# app/translate.py
from functools import lru_cache
import os

import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# ---------------------------
# Env toggles (속도/품질 제어)
# ---------------------------
MT_DEVICE = os.getenv("MT_DEVICE", "cuda").lower()   # "cuda" | "cpu"
USE_GPU = (MT_DEVICE == "cuda") and torch.cuda.is_available()
DEVICE_INDEX = 0 if USE_GPU else -1
DEVICE = torch.device(f"cuda:{DEVICE_INDEX}") if USE_GPU else torch.device("cpu")

# 성능/속도 스위치
MT_NUM_BEAMS = int(os.getenv("MT_NUM_BEAMS", "1"))             # 기본 1=greedy (빔↑ → 품질↑/속도↓)
MT_MAX_NEW_TOKENS = int(os.getenv("MT_MAX_NEW_TOKENS", "96"))  # 문장당 생성 상한
MT_LEN_PEN = float(os.getenv("MT_LEN_PEN", "1.0"))
MT_MAX_BATCH_TOKENS = int(os.getenv("MT_MAX_BATCH_TOKENS", "2048"))
MT_MAX_BATCH_SIZE = int(os.getenv("MT_MAX_BATCH_SIZE", "16"))
MT_FAST_ONLY = os.getenv("MT_FAST_ONLY", "1") == "1"           # 빠른 모델만 사용

# (옵션) Hugging Face 토큰
HF_TOKEN = (
    os.getenv("HF_TOKEN")
    or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    or os.getenv("HUGGINGFACE_HUB_TOKEN")
    or None
)

PIPE_KW = {}
if DEVICE_INDEX >= 0:
    PIPE_KW["device"] = DEVICE_INDEX
if HF_TOKEN:
    PIPE_KW["token"] = HF_TOKEN

HF_AUTH_KW = {}
if HF_TOKEN:
    HF_AUTH_KW["use_auth_token"] = HF_TOKEN

# TF32 가속 (Ampere+)
if torch.cuda.is_available():
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

# ---------------------------
# 모델 우선순위 (가벼운 → 무거운)
#   * ja↔en 단일-hop 우선
#   * en↔ko는 m2m100_418M 우선 (속도)
# ---------------------------
MODEL_PREFS = {
    # en->ko
    ("en", "ko"): (
        ["facebook/m2m100_418M", "facebook/m2m100_1.2B"]
        + ([] if MT_FAST_ONLY else ["google/madlad400-3b-mt"])
        + ["Helsinki-NLP/opus-mt-en-ko"]
    ),
    # ko->en
    ("ko", "en"): ["facebook/m2m100_418M", "Helsinki-NLP/opus-mt-ko-en"],

    # ja<->en (직통)
    ("ja", "en"): [
        "facebook/m2m100_418M",
        "facebook/nllb-200-distilled-600M",
        "Helsinki-NLP/opus-mt-ja-en",
        "Helsinki-NLP/opus-mt-jap-en",
    ],
    ("en", "ja"): [
        "facebook/m2m100_418M",
        "facebook/nllb-200-distilled-600M",
        "Helsinki-NLP/opus-mt-en-jap",
    ],
       ("ko", "ja"): [
        "facebook/m2m100_418M",
        "facebook/nllb-200-distilled-600M",
        "facebook/mbart-large-50-many-to-many-mmt",
    ],
    ("ja", "ko"): [
        "facebook/m2m100_418M",
        "facebook/nllb-200-distilled-600M",
        "facebook/mbart-large-50-many-to-many-mmt",
    ],
}

# ---------------------------
# Pipeline 경로 (fallback)
# ---------------------------
@lru_cache(maxsize=16)
def _get_mt(model_name: str, token_key: int = 1):
    kwargs = dict(PIPE_KW)
    if token_key == 0 and "token" in kwargs:
        kwargs = dict(kwargs)
        kwargs.pop("token", None)
    try:
        return pipeline("translation", model=model_name, **kwargs)
    except Exception as e:
        if token_key == 1:
            try:
                return _get_mt(model_name, token_key=0)
            except Exception:
                pass
        raise

def _run_pipe(mt, texts):
    outs = mt(
        texts,
        max_length=512,
        do_sample=False,
        num_beams=max(1, MT_NUM_BEAMS),
        clean_up_tokenization_spaces=True,
    )
    return [o["translation_text"] for o in outs]

# ---------------------------
# Seq2Seq 직접 로딩 (빠름/유연)
# ---------------------------
@lru_cache(maxsize=4)
def _get_seq2seq(model_name: str):
    tok = AutoTokenizer.from_pretrained(model_name, **HF_AUTH_KW)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(model_name, **HF_AUTH_KW)
    if USE_GPU:
        try:
            mdl = mdl.to(DEVICE, dtype=torch.float16)
        except Exception:
            mdl = mdl.to(DEVICE)
    else:
        mdl = mdl.to(DEVICE)
    mdl.eval()
    mdl._name = model_name
    return tok, mdl

def _estimate_tok_len(tok, s: str) -> int:
    try:
        return len(tok(s, add_special_tokens=False, return_attention_mask=False)["input_ids"])
    except Exception:
        return max(1, len(s) // 2)

def _make_batches_by_tokens(tok, texts):
    idxs = list(range(len(texts)))
    lens = [_estimate_tok_len(tok, texts[i]) for i in idxs]
    order = sorted(idxs, key=lambda i: lens[i])

    batches, i = [], 0
    while i < len(order):
        batch, tok_sum = [], 0
        while (
            i < len(order)
            and len(batch) < MT_MAX_BATCH_SIZE
            and tok_sum + lens[order[i]] <= MT_MAX_BATCH_TOKENS
        ):
            j = order[i]
            batch.append(j)
            tok_sum += lens[j]
            i += 1
        if not batch:
            batch = [order[i]]
            i += 1
        batches.append(batch)
    return batches

def _gen_kwargs():
    return dict(
        max_new_tokens=MT_MAX_NEW_TOKENS,
        num_beams=max(1, MT_NUM_BEAMS),
        do_sample=False,
        length_penalty=MT_LEN_PEN,
        early_stopping=True,
        use_cache=True,
    )

def _run_seq2seq(model_name, texts, src, tgt):
    if not texts:
        return []
    tok, mdl = _get_seq2seq(model_name)
    device = next(mdl.parameters()).device
    out = [None] * len(texts)
    batches = _make_batches_by_tokens(tok, texts)
    kwargs = _gen_kwargs()

    nllb_map = {"en": "eng_Latn", "ko": "kor_Hang", "ja": "jpn_Jpan"}

    with torch.inference_mode():
        for batch in batches:
            xs = [texts[i] for i in batch]
            if "m2m100" in model_name:
                tok.src_lang = src
                enc = tok(xs, return_tensors="pt", padding=True, truncation=True)
                enc = {k: v.to(device) for k, v in enc.items()}
                bos = tok.get_lang_id(tgt)
                gen = mdl.generate(**enc, forced_bos_token_id=bos, **kwargs)
                ys = tok.batch_decode(gen, skip_special_tokens=True)
            elif "nllb" in model_name:
                tok.src_lang = nllb_map.get(src, "eng_Latn")
                enc = tok(xs, return_tensors="pt", padding=True, truncation=True)
                enc = {k: v.to(device) for k, v in enc.items()}
                bos = tok.convert_tokens_to_ids(nllb_map.get(tgt, "kor_Hang"))
                gen = mdl.generate(**enc, forced_bos_token_id=bos, **kwargs)
                ys = tok.batch_decode(gen, skip_special_tokens=True)
            elif "madlad" in model_name:
                enc = tok(xs, return_tensors="pt", padding=True, truncation=True)
                enc = {k: v.to(device) for k, v in enc.items()}
                gen = mdl.generate(**enc, **kwargs)
                ys = tok.batch_decode(gen, skip_special_tokens=True)
            else:
                ys = _run_pipe(_get_mt(model_name), xs)

            for j, text in zip(batch, ys):
                out[j] = text
    return out

# ---------------------------
# 2-hop (최후 폴백용)
# ---------------------------
def _two_hop(texts, src: str, mid: str, tgt: str):
    first_models = MODEL_PREFS.get((src, mid), [])
    if not first_models:
        raise RuntimeError(f"No model for hop {src}->{mid}")
    last_err = None
    for m in first_models:
        try:
            t1 = _run_seq2seq(m, texts, src, mid)
            break
        except Exception as e:
            last_err = e
            continue
    else:
        raise last_err or RuntimeError(f"Failed hop {src}->{mid}")

    second_models = MODEL_PREFS.get((mid, tgt), [])
    if not second_models:
        raise RuntimeError(f"No model for hop {mid}->{tgt}")
    last_err = None
    for m in second_models:
        try:
            return _run_seq2seq(m, t1, mid, tgt)
        except Exception as e:
            last_err = e
            continue
    raise last_err or RuntimeError(f"Failed hop {mid}->{tgt}")

# ---------------------------
# Public API
# ---------------------------
def translate_texts(texts, src: str, tgt: str):
    if src == tgt:
        return texts
    models = MODEL_PREFS.get((src, tgt), [])
    last_err = None
    for m in models:
        try:
            out = _run_seq2seq(m, texts, src, tgt)
            if (src, tgt) == ("en", "ko") and any(_looks_garbled_ko(x) for x in out):
                continue
            return out
        except Exception as e:
            last_err = e
            continue
    # 전부 실패 시에만 2-hop
    return _two_hop(texts, src, "en", tgt)

def _looks_garbled_ko(s: str) -> bool:
    if not s:
        return True
    ja_punct = ("、", "。", "・")
    bad_hit = any(p in s for p in ja_punct)
    hangul = sum(0xAC00 <= ord(ch) <= 0xD7A3 for ch in s)
    latin  = sum(('A' <= ch <= 'Z') or ('a' <= ch <= 'z') for ch in s)
    digits = sum(ch.isdigit() for ch in s)
    total  = len(s)
    return bad_hit or (hangul / max(1, total) < 0.15 and (latin + digits) / max(1, total) > 0.6)
