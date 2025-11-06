# app/main.py
import os

# 🔴 TTS를 CPU로 돌릴 땐 cuDNN 차단
if os.getenv("TTS_DEVICE", "cpu").lower() == "cpu":
    import torch
    torch.backends.cudnn.enabled = False

# ✅ torchaudio soundfile 백엔드
try:
    import torchaudio
    torchaudio.set_audio_backend("soundfile")
except Exception as e:
    print("WARN: torchaudio.set_audio_backend('soundfile') failed:", e)

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional

app = FastAPI(title="AI Dub Prototype", version="0.4.0", servers=[{"url": "/"}])

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🔁 변경: 새 TTS 단계 import
from .pipeline import (
    dub, asr_only, translate_stage,
    tts_probe_stage, tts_finalize_stage,
    mux_stage, merge_segments_stage,
    build_voice_sample_stage, synthesize_single_text,
)
from .utils_meta import load_meta, save_meta
import shutil

# ---------- 바디 모델 (PATCH는 사용 안 해도 OK) ----------
class SegmentPatch(BaseModel):
    i: int
    text: str

class SegmentsPatch(BaseModel):
    items: list[SegmentPatch]

class TranslateBody(BaseModel):
    src: str
    tgt: str
    length_mode: Optional[str] = Field(default="off", description="'off' only (no auto-edit)")

class MergeBody(BaseModel):
    merges: Optional[list[list[int]]] = Field(
        default=None,
        description="0-based inclusive ranges, e.g., [[1,3],[7,8]]"
    )

class TranslationPatch(BaseModel):
    i: int
    text: Optional[str] = None
    start: Optional[float] = Field(default=None, description="seconds (absolute)")
    end: Optional[float]   = Field(default=None, description="seconds (absolute)")

class TranslationsPatch(BaseModel):
    items: list[TranslationPatch]

# ---------- Health ----------
@app.get("/health")
def health():
    return {"ok": True}

# ---------- 원샷 dub ----------
@app.post("/dub")
async def dub_endpoint(
    file: UploadFile = File(..., description="Video or audio file"),
    target_lang: str = Form(..., description="'en' or 'ja'"),
    ref_voice: Optional[UploadFile] = File(None, description="Optional reference WAV (>=6s)"),
):
    assert target_lang in ("en", "ja", "ko"), "target_lang must be 'en' or 'ja' or 'ko'"
    tmp_dir = "/app/data/tmp"
    os.makedirs(tmp_dir, exist_ok=True)

    in_path = os.path.join(tmp_dir, file.filename)
    with open(in_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    ref_path = None
    if ref_voice is not None:
        ref_path = os.path.join(tmp_dir, ref_voice.filename)
        with open(ref_path, "wb") as f2:
            shutil.copyfileobj(ref_voice.file, f2)

    meta = dub(in_path, target_lang=target_lang, ref_wav=ref_path)
    return JSONResponse({"job_id": meta["job_id"], "output": meta["output"], "workdir": meta["workdir"]})

@app.get("/download/{job_id}")
def download(job_id: str):
    file_path = f"/app/data/{job_id}/output.mp4"
    if not os.path.exists(file_path):
        return JSONResponse({"error": "Not found"}, status_code=404)
    return FileResponse(file_path, media_type="video/mp4", filename=f"dub_{job_id}.mp4")

# ---------- 단계형 ----------
# 1) ASR
@app.post("/asr")
async def asr_endpoint(
    file: UploadFile = File(..., description="Video or audio file"),
):
    meta = await asr_only(file)
    return JSONResponse({"job_id": meta["job_id"], "segments": meta["segments"], "workdir": meta["workdir"]})

# (선택) PATCH 엔드포인트들: 안 써도 됩니다
@app.patch("/segments/{job_id}")
def patch_segments(job_id: str, body: SegmentsPatch):
    workdir = f"/app/data/{job_id}"
    meta = load_meta(workdir)
    if "segments" not in meta:
        return JSONResponse({"error": "No ASR segments for this job"}, status_code=400)

    segs = meta["segments"]
    for item in body.items:
        if 0 <= item.i < len(segs):
            segs[item.i]["text"] = item.text.strip()
    meta["segments"] = segs
    save_meta(workdir, meta)
    return {"ok": True, "segments": segs}

# 2) 번역 (자동 수정 없음)
@app.post("/translate/{job_id}")
def translate_endpoint(job_id: str, body: TranslateBody):
    workdir = f"/app/data/{job_id}"
    meta = load_meta(workdir)
    if "segments" not in meta:
        return JSONResponse({"error": "No ASR segments"}, status_code=400)

    translations = translate_stage(meta["segments"], src=body.src, tgt=body.tgt, length_mode="off")
    meta["translations"] = translations
    meta.setdefault("options", {})["lang"] = {"src": body.src, "tgt": body.tgt}
    save_meta(workdir, meta)
    return {"ok": True, "translations": translations}

@app.patch("/translations/{job_id}")
def patch_translations(job_id: str, body: TranslationsPatch):
    workdir = f"/app/data/{job_id}"
    meta = load_meta(workdir)
    if "translations" not in meta:
        return JSONResponse({"error": "No translations"}, status_code=400)

    trs = meta["translations"]
    dur = meta.get("orig_duration")
    for item in body.items:
        if 0 <= item.i < len(trs):
            if item.text is not None:
                trs[item.i]["text"] = item.text.strip()
            if item.start is not None:
                s = max(0.0, float(item.start))
                if dur is not None: s = min(s, float(dur))
                trs[item.i]["start"] = s
            if item.end is not None:
                e = max(0.0, float(item.end))
                if dur is not None: e = min(e, float(dur))
                trs[item.i]["end"] = e
            if trs[item.i]["end"] <= trs[item.i]["start"]:
                trs[item.i]["end"] = trs[item.i]["start"] + 0.05
    meta["translations"] = trs
    save_meta(workdir, meta)
    return {"ok": True, "translations": trs}

# 3) 1차 TTS: 길이 측정 리포트
@app.post("/tts-probe/{job_id}")
async def tts_probe_endpoint(
    job_id: str,
    target_lang: str = Form(..., description="'en' or 'ja' or 'ko'"),
    ref_voice: Optional[UploadFile] = File(None, description="Optional reference WAV (>=6s)")
):
    out = await tts_probe_stage(job_id, target_lang=target_lang, ref_voice=ref_voice)
    return {"ok": True, **out}

# 4) 2차 TTS: 최종 보정/결합
@app.post("/tts-finalize/{job_id}")
async def tts_finalize_endpoint(
    job_id: str,
    target_lang: str = Form(..., description="'en' or 'ja' or 'ko'"),
    ref_voice: Optional[UploadFile] = File(None, description="Optional reference WAV (>=6s)")
):
    out = await tts_finalize_stage(job_id, target_lang=target_lang, ref_voice=ref_voice)
    return {"ok": True, **out}

# (호환) 예전 /tts → finalize로 동작
@app.post("/tts/{job_id}")
async def tts_compat_endpoint(
    job_id: str,
    target_lang: str = Form(..., description="'en' or 'ja' or 'ko'"),
    ref_voice: Optional[UploadFile] = File(None, description="Optional reference WAV (>=6s)")
):
    out = await tts_finalize_stage(job_id, target_lang=target_lang, ref_voice=ref_voice)
    return {"ok": True, **out}

# 🔊 단일 문장 TTS
@app.post("/tts-single")
async def tts_single_endpoint(
    text: str = Form(..., description="문장 또는 문단"),
    target_lang: str = Form(..., description="'en' or 'ja' or 'ko'"),
    ref_voice: UploadFile = File(..., description="참조 목소리 WAV (6초 이상 권장)"),
):
    result = synthesize_single_text(text=text, target_lang=target_lang, ref_voice=ref_voice)
    return FileResponse(
        result["tts_wav"],
        media_type="audio/wav",
        filename=f"tts_{result['job_id']}.wav",
    )

# 5) MUX
@app.post("/mux/{job_id}")
def mux_endpoint(job_id: str):
    path = mux_stage(job_id)
    return {"ok": True, "output": path}


# 6) 병합
@app.post("/merge/{job_id}")
def merge_endpoint(job_id: str, body: Optional[MergeBody] = None):
    merges = body.merges if body and body.merges else None
    out = merge_segments_stage(job_id, merges=merges)
    return {"ok": True, **out}


# 7) Voice Sample (무음 제거하여 음성만 연결)
@app.post("/voice-sample")
async def voice_sample_endpoint(
    file: UploadFile = File(..., description="Video or audio file (.mp4, .wav, etc.)"),
):
    """
    업로드한 파일에서 배경/잡음 분리 → STT 세그먼트 기반으로 무음 구간 제거 → 음성만 연결한 WAV 반환.
    """
    meta = await asr_only(file)
    job_id = meta["job_id"]
    out = build_voice_sample_stage(job_id)
    return FileResponse(out["voice_sample_wav"], media_type="audio/wav", filename=f"voice_sample_{job_id}.wav")


@app.get("/voice-sample/{job_id}")
def voice_sample_download(job_id: str):
    work = f"/app/data/{job_id}"
    meta = load_meta(work)
    path = meta.get("voice_sample_wav")
    if not path or not os.path.exists(path):
        return JSONResponse({"error": "Voice sample not found for this job"}, status_code=404)
    return FileResponse(path, media_type="audio/wav", filename=f"voice_sample_{job_id}.wav")
