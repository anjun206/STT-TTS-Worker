# AI 더빙 프로토타입 (Docker)

faster-whisper(ASR), Helsinki-NLP MT, Coqui XTTS v2 TTS 위에 구축된 한→영/일 더빙 파이프라인입니다. API 스택과 TTS 엔진은 **서로 다른 컨테이너**에서 실행되지만 **동일한 FastAPI 애플리케이션 코드**를 공유하도록 GPU 워커 환경에 맞춰 준비되어 있습니다.

## Docker 레이아웃

```
.
├── backend/
│   ├── app/                     # FastAPI 애플리케이션(공용 소스)
│   ├── Dockerfile.api           # API/ASR 이미지 정의
│   ├── Dockerfile.tts           # TTS 전용 이미지 정의
│   ├── requirements.api.txt     # API 컨테이너 의존성
│   └── requirements.tts.txt     # TTS 컨테이너 의존성
└── docker-compose.yml           # api + tts 서비스를 함께 구동
```

### Compose 서비스

| 서비스   | 역할                                          | 포트     | 비고                                                  |
| ----- | ------------------------------------------- | ------ | --------------------------------------------------- |
| `api` | REST 요청 처리, ASR/번역/믹싱 수행, 합성은 HTTP로 TTS에 전달 | `8000` | `TTS_URL`이 `tts` 서비스를 가리킴; Whisper/MT는 기본적으로 GPU 사용 |
| `tts` | XTTS v2 엔드포인트 제공(주로 `/tts-single`)          | `9000` | 동일 앱 코드를 공유하며 GPU 합성에 최적화                           |

### 빌드 및 실행

```bash
docker compose build      # 두 이미지를 모두 빌드
docker compose up -d      # api + tts 컨테이너를 백그라운드로 실행
```

대화형 문서는 [http://localhost:8000/docs](http://localhost:8000/docs) 에서 확인할 수 있습니다.

## 의존성 분류

### backend/requirements.api.txt

* `fastapi`, `uvicorn[standard]`, `python-multipart`: 웹 서버 코어 및 multipart 업로드.
* `faster-whisper`, `onnxruntime-gpu`: CUDA 가속 ASR 추론.
* `transformers`, `sentencepiece`, `huggingface_hub`: 번역 모델 실행 및 토크나이저 지원.
* `soundfile`: 중간 WAV 자산 입출력.
* `sacremoses`, `cutlet`, `fugashi`, `unidic-lite`: 번역 단계에서 쓰는 일본어 토크나이징/로마자 표기 도구.
* `requests`: API 컨테이너가 HTTP로 원격 TTS 워커를 호출.
* `webrtcvad`: 무음 트리밍을 위한 음성활동감지(VAD) 폴백.
* `torchcodec`: 파이프라인 내 오디오 조작 유틸리티 참고.

### backend/requirements.tts.txt

* `fastapi`, `uvicorn[standard]`, `python-multipart`: `/tts-single`을 호스팅하는 동일 FastAPI 표면.
* `TTS`: Coqui XTTS v2 합성 라이브러리.
* `soundfile`: 생성된 파형 입출력.
* `requests`: 보조 HTTP 호출 시 API 환경과의 파리티 유지.

## docker-compose 개요

`docker-compose.yml`은 저장소 루트에서 두 이미지를 빌드하고, 라이브 코드 리로드를 위해 `backend/app` 디렉터리를 마운트합니다. 또한 호스트 캐시(`./data/hf_cache`, `./data/tts_cache`, `./data/demucs_cache`)를 공유해 실행 간 모델 다운로드가 유지되도록 합니다. `api` 서비스는 `tts`에 의존하도록 설정되어 합성기가 준비되기 전에 외부 트래픽을 받지 않습니다.

핵심 환경 변수:

* `TTS_URL`: API 컨테이너가 합성을 위임할 TTS 워커 주소(예: `http://tts:9000`).
* `USE_GPU`, `MT_DEVICE`, `TTS_DEVICE`, `NVIDIA_VISIBLE_DEVICES`: CUDA 사용 전환 및 장치 지정(여러 GPU로 워크로드 분산 가능).
* `HF_HOME`, `TRANSFORMERS_CACHE`, `TTS_HOME`, `DEMUCS_CACHE`: 호스트 바인드 마운트 경로와 정렬하여 가중치 재다운로드 방지.
* `MT_*`: 번역 빔 서치/배칭 등을 제어해 속도와 정확도 균형 조정.

## API 빠른 시작

```powershell
# 전체 더빙(한국어 -> 영어)
Invoke-RestMethod -Uri http://localhost:8000/dub -Method Post -Form @{
  file        = Get-Item .\sample.mp4
  target_lang = "en"
}

# 사용자 정의 레퍼런스 보이스로 더빙
Invoke-RestMethod -Uri http://localhost:8000/dub -Method Post -Form @{
  file        = Get-Item .\sample.mp4
  ref_voice   = Get-Item .\ref.wav
  target_lang = "ja"
}
```

응답에는 `job_id`가 포함됩니다. `./data/<job_id>/output.mp4`를 확인하거나 `GET /download/{job_id}`를 호출해 최종(믹스된) 영상을 내려받을 수 있습니다.

## 스토리지 레이아웃

* `./data`: 작업 단위(job)별 워크스페이스(입력, 중간 산출물, 최종 렌더)를 두 컨테이너가 공유.
* `./data/hf_cache`, `./data/tts_cache`, `./data/demucs_cache`: Hugging Face, Coqui XTTS, Demucs 캐시(디스크 회수 시 삭제해도 안전).
