"""
STT 전용 워커

기능 개요
- SQS 큐에서 메시지를 Long Polling으로 수신
- S3에서 입력 비디오/오디오를 다운로드 → ffmpeg로 mono 16kHz WAV 추출
- faster-whisper로 STT 수행
- 프로젝트 버킷 규칙에 따라 자막(SRT/VTT) 및 manifest JSON을 S3에 업로드
- 성공 시 SQS 메시지 삭제(실패 시 삭제하지 않음 → 재시도 또는 DLQ)

메시지 필드(예시)
- task: "stt" 고정 (워커가 STT 작업만 처리)
- projectId, uploadId: 입력 키 자동생성 시 필수
- assetId: 산출물 식별자(없으면 uploadId 또는 UUID 사용)
- targetLang: 출력 폴더 언어 코드(예: "ko")
- inputType: "video" 또는 "audio" (기본 video)
- subtitleFormat: "srt" 또는 "vtt" (기본 srt)
- language: STT 입력 언어 고정(없으면 자동 감지)
- bucket, input_key: 수동으로 입력 키를 지정하고자 할 때 사용

S3 키 규칙(요구사항에 맞춤)
- 입력: projects/{projectId}/inputs/(videos|audios)/{uploadId}.{ext}
- 출력 자막: projects/{projectId}/outputs/{targetLang}/subtitles/{assetId}.srt|vtt
- manifest: projects/{projectId}/manifests/{assetId}.json
"""

# backend/app/worker_stt.py
import os, json, time, tempfile, subprocess, uuid, sys
from typing import Optional, Tuple
import boto3
from botocore.exceptions import BotoCoreError, ClientError
from faster_whisper import WhisperModel

SQS_QUEUE_URL = os.getenv("SQS_QUEUE_URL")
AWS_REGION = os.getenv("AWS_REGION", "ap-northeast-2")
DEFAULT_BUCKET = os.getenv("S3_BUCKET")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "large-v3")
USE_GPU = os.getenv("USE_GPU", "0") == "1"


def log(*args):
    print(*args, flush=True)


def ffmpeg_extract_wav(input_path: str, out_path: str, sr: int = 16000) -> None:
    """ffmpeg로 mono 16kHz WAV 추출

    -ac 1: 모노
    -ar <sr>: 샘플레이트(Hz)
    -vn: 비디오 스트림 제거(오디오만 추출)
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
        "-ac",
        "1",
        "-ar",
        str(sr),
        # -vn: 비디오 스트림 제거(오디오만 추출)
        "-vn",
        "-f",
        "wav",
        out_path,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def transcribe(model: WhisperModel, wav_path: str, language: Optional[str] = None):
    """faster-whisper로 음성 인식 수행

    language가 None이면 자동 감지. VAD 필터 사용으로 무성 구간을 걸러 품질 향상.
    반환 구조: { language, duration, segments:[{start, end, text}...] }
    """
    segments, info = model.transcribe(
        wav_path,
        language=language,  # None이면 자동감지
        vad_filter=True,
        beam_size=1,
        temperature=0.0,
    )
    segs = []
    for s in segments:
        segs.append({"start": float(s.start), "end": float(s.end), "text": s.text})
    return {"language": info.language, "duration": info.duration, "segments": segs}


def _format_ts_srt(sec: float) -> str:
    """SRT 타임스탬프 포맷(, 밀리초 구분)"""
    ms = int(round((sec - int(sec)) * 1000))
    s = int(sec) % 60
    m = (int(sec) // 60) % 60
    h = int(sec) // 3600
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _format_ts_vtt(sec: float) -> str:
    """VTT 타임스탬프 포맷(. 밀리초 구분)"""
    ms = int(round((sec - int(sec)) * 1000))
    s = int(sec) % 60
    m = (int(sec) // 60) % 60
    h = int(sec) // 3600
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def write_subtitles(segments, path: str, fmt: str = "srt") -> None:
    """세그먼트 리스트를 SRT/VTT 파일로 저장"""
    fmt = (fmt or "srt").lower()
    if fmt not in ("srt", "vtt"):
        fmt = "srt"
    with open(path, "w", encoding="utf-8") as f:
        if fmt == "vtt":
            f.write("WEBVTT\n\n")
        for i, seg in enumerate(segments, 1):
            start = seg["start"]
            end = seg["end"]
            text = seg["text"].strip()
            if fmt == "srt":
                f.write(f"{i}\n{_format_ts_srt(start)} --> {_format_ts_srt(end)}\n{text}\n\n")
            else:
                f.write(f"{_format_ts_vtt(start)} --> {_format_ts_vtt(end)}\n{text}\n\n")


def _infer_project_id_from_key(key: str) -> Optional[str]:
    """S3 key에서 projects/{projectId}/ 패턴을 찾아 projectId 추출"""
    try:
        parts = [p for p in key.split("/") if p]
        for i, p in enumerate(parts):
            if p == "projects" and i + 1 < len(parts):
                return parts[i + 1]
    except Exception:
        pass
    return None


def _get_project_id(msg_body: dict) -> str:
    """projectId 우선순위: 명시값 > input_key로부터 추론 > 오류"""
    project_id = msg_body.get("projectId") or msg_body.get("project_id")
    if project_id:
        return project_id
    key = msg_body.get("input_key")
    if key:
        inferred = _infer_project_id_from_key(key)
        if inferred:
            return inferred
    raise ValueError("projectId is required to build output keys")


def build_input_key(msg_body: dict) -> Tuple[str, str]:
    """입력 S3 키 생성

    - input_key가 명시된 경우 그대로 사용
    - 아니면 프로젝트 규칙에 맞춰 자동 생성
    반환: (key, 확장자)
    """
    # 우선순위: input_key 명시 시 그대로 사용
    if "input_key" in msg_body and msg_body.get("input_key"):
        return msg_body["input_key"], os.path.splitext(msg_body["input_key"])[1].lstrip(".")

    # 버킷 구조 고정값 생성: projects/{projectId}/inputs/(videos|audios)/{uploadId}.{ext}
    project_id = msg_body.get("projectId") or msg_body.get("project_id")
    upload_id = msg_body.get("uploadId") or msg_body.get("upload_id")
    input_type = (msg_body.get("inputType") or msg_body.get("input_type") or "video").lower()
    input_ext = (msg_body.get("inputExt") or msg_body.get("input_ext") or ("mp4" if input_type == "video" else "mp3")).lower()
    if not project_id or not upload_id:
        raise ValueError("projectId and uploadId are required when input_key is not provided")
    if input_type not in ("video", "audio", "videos", "audios"):
        input_type = "video"
    folder = "videos" if input_type.startswith("video") else "audios"
    key = f"projects/{project_id}/inputs/{folder}/{upload_id}.{input_ext}"
    return key, input_ext


def build_output_keys(msg_body: dict, detected_lang: str, subtitle_fmt: str) -> Tuple[str, str]:
    """출력 자막/매니페스트 S3 키 생성

    - 자막: outputs/{targetLang}/subtitles/{assetId}.srt|vtt
    - 매니페스트: manifests/{assetId}.json
    """
    # outputs/{targetLang}/subtitles/{assetId}.srt + manifests/{assetId}.json
    project_id = _get_project_id(msg_body)
    target_lang = (msg_body.get("targetLang") or msg_body.get("target_lang") or detected_lang or "und")
    asset_id = msg_body.get("assetId") or msg_body.get("asset_id") or msg_body.get("uploadId") or str(uuid.uuid4())
    # _get_project_id에서 검증됨
    sub_ext = "vtt" if subtitle_fmt.lower() == "vtt" else "srt"
    subtitle_key = f"projects/{project_id}/outputs/{target_lang}/subtitles/{asset_id}.{sub_ext}"
    manifest_key = f"projects/{project_id}/manifests/{asset_id}.json"
    return subtitle_key, manifest_key


def process_message(s3, msg_body: dict, model: WhisperModel):
    """단일 메시지 처리 파이프라인

    1) 입력 키 확정 및 다운로드
    2) ffmpeg로 오디오 추출(WAV)
    3) STT 수행
    4) 자막 파일 생성(SRT/VTT)
    5) 자막/매니페스트 업로드
    """
    task = msg_body.get("task")
    if task != "stt":
        return {"skipped": True, "reason": f"unsupported task: {task}"}

    job_id = msg_body.get("job_id") or str(uuid.uuid4())
    bucket = msg_body.get("bucket") or DEFAULT_BUCKET
    subtitle_fmt = (msg_body.get("subtitleFormat") or msg_body.get("subtitle_format") or "srt").lower()
    force_lang = msg_body.get("language")  # STT 입력 언어(없으면 자동)
    if not bucket:
        raise ValueError("bucket is required")

    # 입력 키 결정(구조 자동생성 or 직접 지정)
    input_key, input_ext = build_input_key(msg_body)

    with tempfile.TemporaryDirectory() as td:
        src_in = os.path.join(td, f"input.{input_ext}")
        src_wav = os.path.join(td, "audio.wav")
        out_subs = os.path.join(td, f"subtitle.{ 'vtt' if subtitle_fmt=='vtt' else 'srt'}")
        out_manifest = os.path.join(td, "manifest.json")

        # 1) 다운로드
        log(f"[{job_id}] Download s3://{bucket}/{input_key}")
        s3.download_file(bucket, input_key, src_in)

        # 2) 오디오 추출
        log(f"[{job_id}] Extract WAV via ffmpeg")
        ffmpeg_extract_wav(src_in, src_wav)

        # 3) STT (language 미지정 시 자동 감지)
        log(f"[{job_id}] Transcribe with model={WHISPER_MODEL}, lang={force_lang}")
        result = transcribe(model, src_wav, language=force_lang)

        # 4) 로컬 저장 + (옵션) S3 업로드
        # 4) 자막(SRT/VTT) 생성
        write_subtitles(result["segments"], out_subs, fmt=subtitle_fmt)

        # 5) 출력 키 결정 및 업로드
        subtitle_key, manifest_key = build_output_keys(msg_body, result.get("language"), subtitle_fmt)
        log(f"[{job_id}] Upload subtitles to s3://{bucket}/{subtitle_key}")
        s3.upload_file(out_subs, bucket, subtitle_key)

        # projectId는 명시값 없을 시 input_key에서 추론
        project_id_for_manifest = _get_project_id(msg_body)

        manifest = {
            "assetId": msg_body.get("assetId") or msg_body.get("asset_id") or msg_body.get("uploadId") or job_id,
            "projectId": project_id_for_manifest,
            "uploadId": msg_body.get("uploadId") or msg_body.get("upload_id"),
            "task": "stt",
            "input": {"bucket": bucket, "key": input_key},
            "output": {"bucket": bucket, "subtitle": subtitle_key},
            "meta": {
                "detectedLanguage": result.get("language"),
                "duration": result.get("duration"),
                "segments": len(result.get("segments", [])),
            },
        }
        with open(out_manifest, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False)

        log(f"[{job_id}] Upload manifest to s3://{bucket}/{manifest_key}")
        s3.upload_file(out_manifest, bucket, manifest_key)

        return {"ok": True, "job_id": job_id, "subtitle_key": subtitle_key, "manifest_key": manifest_key}


def main():
    """워커 진입점: SQS 롱 폴링 → 메시지 처리 루프"""
    if not SQS_QUEUE_URL:
        log("SQS_QUEUE_URL not set")
        sys.exit(1)

    session = boto3.Session(region_name=AWS_REGION)
    sqs = session.client("sqs")
    s3 = session.client("s3")

    device = "cuda" if USE_GPU else "cpu"
    # GPU 사용 시 float16, CPU 사용 시 int8로 메모리 사용량/속도 밸런스 확보
    compute_type = "float16" if USE_GPU else "int8"
    log(f"Loading Whisper model={WHISPER_MODEL} on {device} ({compute_type})")
    model = WhisperModel(WHISPER_MODEL, device=device, compute_type=compute_type)

    log("Worker started. Polling SQS...")
    while True:
        try:
            resp = sqs.receive_message(
                QueueUrl=SQS_QUEUE_URL,
                MaxNumberOfMessages=1,
                WaitTimeSeconds=20,  # long polling
                VisibilityTimeout=120,  # 처리시간 여유
            )
            msgs = resp.get("Messages", [])
            if not msgs:
                continue

            msg = msgs[0]
            receipt = msg["ReceiptHandle"]
            body_raw = msg.get("Body", "{}")
            try:
                body = json.loads(body_raw)
            except json.JSONDecodeError:
                log("Invalid JSON message:", body_raw)
                # DLQ 없으면 그냥 삭제는 위험. 여기서는 건너뜀(visibility timeout 후 재전송)
                time.sleep(1)
                continue

            try:
                result = process_message(s3, body, model)
                log("Processed:", result)
                # 성공 시 삭제
                sqs.delete_message(QueueUrl=SQS_QUEUE_URL, ReceiptHandle=receipt)
            except (
                BotoCoreError,
                ClientError,
                subprocess.CalledProcessError,
                Exception,
            ) as e:
                log("Error processing message:", repr(e))
                # 실패 시 삭제하지 않음 → 가시성 타임아웃 후 재전송되거나 DLQ로 이동
                time.sleep(1)

        except KeyboardInterrupt:
            log("Shutting down worker")
            break
        except Exception as e:
            log("Poll error:", repr(e))
            time.sleep(2)


if __name__ == "__main__":
    main()
