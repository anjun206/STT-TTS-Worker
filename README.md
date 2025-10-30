
# AI 더빙 프로토타입 (Docker, GPU 워커)

faster-whisper(ASR), Transformers MT, Coqui XTTS v2(TTS)로 구성된 한→영/일 더빙 파이프라인입니다.  
API 스택과 TTS 엔진을 **서로 다른 컨테이너**로 분리해 GPU 리소스를 유연하게 확장/분배할 수 있습니다.

## 요구사항
- Docker 24+, docker compose v2
- NVIDIA Driver + **nvidia-container-toolkit**
- (Windows) WSL2 + WSL용 CUDA 드라이버

## 레이아웃

```

.
├─ backend/
│  ├─ app/                    # FastAPI 애플리케이션(공용 소스)
│  ├─ Dockerfile.api          # API/ASR 이미지
│  ├─ Dockerfile.tts          # TTS 이미지
│  ├─ requirements.api.txt    # API 의존성
│  └─ requirements.tts.txt    # TTS 의존성
├─ data/                      # 작업/캐시 볼륨(커밋 금지)
└─ deploy/
└─ docker-compose.yml      # api + tts 묶음 실행

````

> ⚠️ 모델/캐시는 커밋하지 마세요. (`data/hf_cache`, `data/tts_cache`, `data/demucs_cache`는 .gitignore)

## 빠른 시작

```bash
cd deploy
docker compose up -d --build

# 스모크 테스트
curl http://localhost:8000/health   # API 워커
curl http://localhost:9000/health   # TTS 워커
````

대화형 문서는 [http://localhost:8000/docs](http://localhost:8000/docs) 에서 확인할 수 있습니다.

### 빌드/실행(부분)

```bash
cd deploy
docker compose build           # 둘 다 빌드
docker compose build api       # API만
docker compose build tts       # TTS만
docker compose up -d api       # API만 실행
docker compose up -d tts       # TTS만 실행
docker compose logs -f         # 로그 팔로우
```

> 루트에서 실행하려면 `docker compose -f deploy/docker-compose.yml up -d --build` 처럼 `-f`를 사용하세요.

## 핵심 환경변수

* **`TTS_URL`**: API 컨테이너가 합성을 위임할 TTS 주소 (내부 네트워크 기준, 기본 `http://tts:9000`)
* **`USE_GPU` / `MT_DEVICE` / `TTS_DEVICE`**: GPU 사용 전환. GPU 분리 예:

  ```yaml
  services:
    api: { gpus: "device=0" }
    tts: { gpus: "device=1" }
  ```
* **캐시**: `HF_HOME`, `TRANSFORMERS_CACHE`, `TTS_HOME`, `DEMUCS_CACHE` → `./data/**` 볼륨과 매칭되어 모델 재다운로드 방지
* **번역 튜닝**: `MT_FAST_ONLY`, `MT_NUM_BEAMS`, `MT_MAX_NEW_TOKENS`, `MT_MAX_BATCH_*`

> NVIDIA 드라이버 + `nvidia-container-toolkit` 필요

## 의존성 분류

### backend/requirements.api.txt

* `fastapi`, `uvicorn[standard]`, `python-multipart`: API/업로드
* `faster-whisper`, `onnxruntime-gpu`: ASR/CUDA 추론
* `transformers`, `sentencepiece`, `huggingface_hub`: 번역 모델/토크나이저
* `webrtcvad`, `demucs`: VAD, 보컬/배경 분리
* `soundfile`, `requests`, `sacremoses`, `cutlet`, `fugashi`, `unidic-lite`, `torchcodec`

### backend/requirements.tts.txt

* `fastapi`, `uvicorn[standard]`, `python-multipart`
* `TTS`(Coqui XTTS v2), `soundfile`, `requests`

  > **TTS 전용 이미지**라면 `faster-whisper` 등은 굳이 필요 없습니다. 최소 구성을 유지하세요.

## 스토리지(볼륨)

* `./data` : job별 작업/출력 공유
* `./data/hf_cache`, `./data/tts_cache`, `./data/demucs_cache` : 모델 캐시

## 트러블슈팅

* **포트 충돌(8000/9000)**: 기존 컨테이너가 점유 → 정리 후 재시작 (`docker ps`, `docker stop/rm`)
* **GPU 미인식**: `docker run --gpus all nvidia/cuda:12.8.0-base nvidia-smi`로 먼저 점검
* **캐시 권한**: 리눅스 호스트에서 권한 문제면 `sudo chown -R $USER:$USER data` 후 재시작
