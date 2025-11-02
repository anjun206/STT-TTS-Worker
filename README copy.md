# AI Dubbing Prototype (Docker)

한국어 영상 → 영어/일본어 더빙 **프로토타입**입니다.  
파이프라인: **FFmpeg 추출 → faster-whisper(STT) → Helsinki-NLP 번역 → Coqui XTTS v2(보이스 클로닝 TTS) → 타임스트레치/컨캣 → 영상에 오디오 교체**

## ⚙️ 요구사항
- Docker Desktop (Windows 가능)
- 최초 실행 시 모델 다운로드로 시간이 걸릴 수 있습니다.

## 🚀 실행
```bash
# 프로젝트 루트에서
docker compose build
docker compose up
```

서버: http://localhost:8000/docs

## 🔊 Voice Sample (무음 제거 음성 추출)
- Swagger에서 `POST /voice-sample` 엔드포인트로 `.mp4/.wav` 업로드
- 처리: BGM/잡음 분리 → STT 세그먼트로 무음 구간 제거 → 음성만 연결한 WAV 반환
- 생성물은 `./data/<job_id>/voice_sample_24k.wav` 에도 저장됩니다.

예시 (PowerShell):
```powershell
Invoke-WebRequest -Uri http://localhost:8000/voice-sample -Method Post -Form @{
  file = Get-Item .\sample.mp4
} -OutFile voice_sample.wav
```

## 🧪 사용 예시 (PowerShell)
```powershell
# 영어 더빙
Invoke-RestMethod -Uri http://localhost:8000/dub -Method Post -Form @{
  file = Get-Item .\sample.mp4
  target_lang = "en"
}

# 일본어 더빙 + 레퍼런스(원 화자 6초 이상 WAV)
Invoke-RestMethod -Uri http://localhost:8000/dub -Method Post -Form @{
  file = Get-Item .\sample.mp4
  ref_voice = Get-Item .\ref.wav
  target_lang = "ja"
}
```

응답 JSON의 `output` 경로는 컨테이너 내부 경로이므로, 로컬에서는 `./data/<job_id>/output.mp4` 로 접근하세요.  
또는 `GET /download/{job_id}` 로 직접 다운로드 할 수 있습니다.

## 📦 볼륨/캐시
- `./data` : 입력/출력 작업 폴더 (호스트에 그대로 저장)
- `hf_cache` : Hugging Face 캐시(모델)
- `tts_cache` : Coqui TTS 캐시(모델)

## 📝 참고
- STT: faster-whisper (CPU, int8)
- 번역: Helsinki-NLP/opus-mt-* (ko↔en, en↔ja 등)
- TTS: Coqui **XTTS v2** (멀티링구얼 + 보이스 클로닝)

> 품질을 올리려면: 화자 분리/참고 음성 추출(VAD/diarization), 용어집 기반 후편집, MFA(Forced Alignment) 추가 등을 고려하세요.
