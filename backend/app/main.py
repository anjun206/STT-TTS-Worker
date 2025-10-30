# 최소 FastAPI 스켈레톤: /health, /tts-single(더미)
# - api 컨테이너와 tts 컨테이너가 동일 파일을 사용해도 돌아가도록 구성
import os
import tempfile
import wave
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse

app = FastAPI(title="GPU Worker", version="0.0.1", servers=[{"url": "/"}])

@app.get("/health")
def health():
    return {"ok": True, "role": os.getenv("ROLE", "generic")}

def _make_silence(path: str, seconds: float = 0.5, sr: int = 24000):
    # 간단한 16-bit PCM 모노 무음 WAV 생성 (테스트용)
    frames = b"\x00\x00" * int(sr * seconds)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
        wf.writeframes(frames)

@app.post("/tts-single")
async def tts_single(
    text: str = Form(...),
    target_lang: str = Form(...),
    ref_voice: UploadFile = File(...)
):
    # 테스트용: 진짜 합성 대신 0.5초 무음 반환
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    _make_silence(tmp.name, seconds=0.5)
    return FileResponse(tmp.name, media_type="audio/wav", filename="tts.wav")

# 선택: API 컨테이너에서 TTS에 프록시 호출 확인용 엔드포인트
@app.get("/probe-tts")
def probe_tts():
    import requests
    tts_url = os.getenv("TTS_URL")
    if not tts_url:
        return JSONResponse({"error": "TTS_URL not set"}, status_code=500)
    files = {"ref_voice": ("ref.wav", b"\x00\x00" * 48000, "audio/wav")}
    data = {"text": "hello", "target_lang": "en"}
    r = requests.post(tts_url.rstrip("/") + "/tts-single", files=files, data=data, timeout=30)
    return {"ok": r.status_code == 200, "status": r.status_code}
