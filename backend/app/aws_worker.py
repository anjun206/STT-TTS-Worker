"""
AWS 연동용 SQS → 파이프라인 자동화 워커.

주요 흐름
- SQS 롱 폴링으로 작업 메시지 수신
- S3에서 입력 미디어 다운로드
- pipeline.py 모듈을 통해 ASR → 번역 → TTS → 믹스 실행
- 산출물(자막, 더빙 오디오, 최종 영상, manifest)을 S3에 업로드
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from fastapi import UploadFile

from .pipeline import (
    asr_only,
    mux_stage,
    tts_finalize_stage,
    translate_stage,
)
from .utils_meta import load_meta, save_meta

# --------------------------------------------------------------------------- #
# 환경 변수

AWS_REGION = os.getenv("AWS_REGION", "ap-northeast-2")
SQS_QUEUE_URL = os.getenv("SQS_QUEUE_URL")
DEFAULT_BUCKET = os.getenv("S3_BUCKET")
DEFAULT_SUBTITLE_FORMAT = os.getenv("DEFAULT_SUBTITLE_FORMAT", "srt").lower()
ENABLE_WORKER = os.getenv("ENABLE_SQS_WORKER", "0") == "1"


def log(*args):
    print("[AWS]", *args, flush=True)


# --------------------------------------------------------------------------- #
# 자막 작성 유틸

def _format_ts_srt(sec: float) -> str:
    ms = int(round((sec - int(sec)) * 1000))
    s = int(sec) % 60
    m = (int(sec) // 60) % 60
    h = int(sec) // 3600
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _format_ts_vtt(sec: float) -> str:
    ms = int(round((sec - int(sec)) * 1000))
    s = int(sec) % 60
    m = (int(sec) // 60) % 60
    h = int(sec) // 3600
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def _write_subtitles(segments, path: str, fmt: str) -> None:
    fmt_norm = (fmt or "srt").lower()
    if fmt_norm not in ("srt", "vtt"):
        fmt_norm = "srt"

    with open(path, "w", encoding="utf-8") as f:
        if fmt_norm == "vtt":
            f.write("WEBVTT\n\n")
        for idx, seg in enumerate(segments, 1):
            start = float(seg["start"])
            end = float(seg["end"])
            text = seg["text"].strip()
            if fmt_norm == "srt":
                f.write(f"{idx}\n{_format_ts_srt(start)} --> {_format_ts_srt(end)}\n{text}\n\n")
            else:
                f.write(f"{_format_ts_vtt(start)} --> {_format_ts_vtt(end)}\n{text}\n\n")


# --------------------------------------------------------------------------- #
# S3 경로 구성

def _build_input_key(body: dict) -> Tuple[str, str]:
    """
    입력 키를 message body로부터 추론한다.
    Returns (key, extension)
    """
    explicit = body.get("input_key") or body.get("inputKey")
    if explicit:
        return explicit, os.path.splitext(explicit)[1].lstrip(".")

    project_id = body.get("projectId") or body.get("project_id")
    upload_id = body.get("uploadId") or body.get("upload_id")
    if not project_id or not upload_id:
        raise ValueError("projectId and uploadId are required")

    input_type = (body.get("inputType") or body.get("input_type") or "video").lower()
    input_ext = (body.get("inputExt") or body.get("input_ext") or ("mp4" if input_type.startswith("video") else "mp3")).lower()
    folder = "videos" if input_type.startswith("video") else "audios"
    key = f"projects/{project_id}/inputs/{folder}/{upload_id}.{input_ext}"
    return key, input_ext


def _build_output_keys(project_id: str, target_lang: str, asset_id: str, subtitle_fmt: str) -> Dict[str, str]:
    target_lang = target_lang.lower()
    fmt = "vtt" if subtitle_fmt.lower() == "vtt" else "srt"
    base = f"projects/{project_id}/outputs/{target_lang}"
    return {
        "subtitle": f"{base}/subtitles/{asset_id}.{fmt}",
        "audio": f"{base}/audio/{asset_id}.wav",
        "video": f"{base}/video/{asset_id}.mp4",
        "manifest": f"projects/{project_id}/manifests/{asset_id}.json",
    }


# --------------------------------------------------------------------------- #
# 파이프라인 실행

async def _run_pipeline(local_path: str, target_lang: str, source_lang: str, subtitle_fmt: str) -> Dict:
    """
    로컬 파일을 대상으로 파이프라인 전체(ASR→번역→TTS→MUX)를 수행한다.
    Returns dict with paths and metadata.
    """
    filename = os.path.basename(local_path)
    # UploadFile 래퍼를 씌워 FastAPI 엔드포인트가 호출하는 기존 파이프라인 함수(asr_only)를 그대로 재사용한다.
    upload = UploadFile(filename=filename, file=open(local_path, "rb"))
    try:
        meta = await asr_only(upload)
    finally:
        try:
            await upload.close()
        except Exception:
            pass

    job_id = meta["job_id"]
    workdir = meta["workdir"]

    # 번역 결과를 메타에 저장해 TTS 단계가 동일한 데이터 구조를 사용할 수 있게 한다.
    translations = translate_stage(meta["segments"], src=source_lang, tgt=target_lang, length_mode="off")
    meta["translations"] = translations
    meta.setdefault("options", {})["lang"] = {"src": source_lang, "tgt": target_lang}
    save_meta(workdir, meta)

    await tts_finalize_stage(job_id, target_lang=target_lang, ref_voice=None)
    mux_stage(job_id)
    refreshed = load_meta(workdir)

    subtitle_ext = "vtt" if subtitle_fmt.lower() == "vtt" else "srt"
    subtitle_path = os.path.join(workdir, f"subtitle.{subtitle_ext}")
    _write_subtitles(translations, subtitle_path, subtitle_fmt)

    result = {
        "job_id": job_id,
        "workdir": workdir,
        "subtitle_path": subtitle_path,
        "video_path": refreshed.get("output"),
        "audio_path": refreshed.get("dubbed_wav"),
        "meta": refreshed,
    }
    return result


# --------------------------------------------------------------------------- #
# 워커 구현

class SkipMessage(Exception):
    """처리하지 않을 메시지를 표기하기 위한 내부 예외."""


@dataclass
class WorkerConfig:
    bucket: str
    subtitle_fmt: str
    target_lang: str
    source_lang: str
    project_id: str
    upload_id: str
    asset_id: str
    input_key: str
    input_ext: str


def _normalize_message(body: dict) -> WorkerConfig:
    task = (body.get("task") or "dub").lower()
    if task not in ("dub", "ai-dub", "pipeline"):
        raise SkipMessage(f"unsupported task: {task}")

    bucket = body.get("bucket") or DEFAULT_BUCKET
    if not bucket:
        raise ValueError("bucket is required")

    project_id = body.get("projectId") or body.get("project_id")
    upload_id = body.get("uploadId") or body.get("upload_id")
    asset_id = body.get("assetId") or body.get("asset_id") or upload_id or uuid.uuid4().hex
    target_lang = (body.get("targetLang") or body.get("target_lang") or "").lower()
    if not project_id or not upload_id or not target_lang:
        raise ValueError("projectId, uploadId, targetLang are required")

    source_lang = (body.get("sourceLang") or body.get("source_lang") or body.get("language") or "ko").lower()
    subtitle_fmt = (body.get("subtitleFormat") or body.get("subtitle_format") or DEFAULT_SUBTITLE_FORMAT).lower()

    # 메시지에 input_key가 없으면 표준 경로 규칙(projects/{id}/inputs/...)으로 대체한다.
    input_key, input_ext = _build_input_key(body)
    return WorkerConfig(
        bucket=bucket,
        subtitle_fmt=subtitle_fmt,
        target_lang=target_lang,
        source_lang=source_lang,
        project_id=project_id,
        upload_id=upload_id,
        asset_id=asset_id,
        input_key=input_key,
        input_ext=input_ext,
    )


class AWSLongPollingWorker:
    def __init__(self):
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    # ------------------------------------------------------------------ #
    def start(self):
        if self._thread and self._thread.is_alive():
            return
        if not SQS_QUEUE_URL:
            log("SQS_QUEUE_URL not set; AWS worker disabled.")
            return
        if not DEFAULT_BUCKET:
            log("S3_BUCKET not set; AWS worker disabled.")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="aws-sqs-worker", daemon=True)
        self._thread.start()
        log("background worker started.")

    def stop(self, timeout: float = 5.0):
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
            log("background worker stopped.")

    def is_running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    # ------------------------------------------------------------------ #
    def _run(self):
        session = boto3.Session(region_name=AWS_REGION)
        sqs = session.client("sqs")
        s3 = session.client("s3")
        # FastAPI 메인 루프와 격리된 전용 이벤트 루프를 생성해 백그라운드 스레드에서 실행한다.
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        log(f"Polling SQS queue {SQS_QUEUE_URL}")

        while not self._stop_event.is_set():
            try:
                resp = sqs.receive_message(
                    QueueUrl=SQS_QUEUE_URL,
                    MaxNumberOfMessages=1,
                    WaitTimeSeconds=20,
                    VisibilityTimeout=900,
                )
            except Exception as e:
                log(f"SQS receive error: {e!r}")
                time.sleep(5)
                continue

            messages = resp.get("Messages", [])
            if not messages:
                continue

            msg = messages[0]
            receipt = msg["ReceiptHandle"]
            body_raw = msg.get("Body", "{}")
            try:
                payload = json.loads(body_raw)
            except json.JSONDecodeError:
                log("Invalid JSON payload; message buried.")
                sqs.delete_message(QueueUrl=SQS_QUEUE_URL, ReceiptHandle=receipt)
                continue

            try:
                config = _normalize_message(payload)
            except SkipMessage as skip:
                log(skip)
                sqs.delete_message(QueueUrl=SQS_QUEUE_URL, ReceiptHandle=receipt)
                continue
            except Exception as e:
                log(f"Message rejected: {e}")
                time.sleep(1)
                continue

            try:
                # 동기식 SQS 루프에서 비동기 파이프라인 처리를 동기적으로 기다린다.
                result = self._loop.run_until_complete(
                    self._process_job(s3, config)
                )
                log(f"Processed job {result.get('job_id')} (asset {config.asset_id})")
                sqs.delete_message(QueueUrl=SQS_QUEUE_URL, ReceiptHandle=receipt)
            except (BotoCoreError, ClientError) as e:
                log(f"AWS client error: {e!r}")
                time.sleep(5)
            except Exception as e:
                log(f"Processing error: {e!r}")
                time.sleep(5)

        asyncio.set_event_loop(None)

    # ------------------------------------------------------------------ #
    async def _process_job(self, s3_client, config: WorkerConfig) -> Dict:
        # 각 메시지 별로 임시 디렉터리를 생성해 산출물을 안전하게 관리한다.
        with tempfile.TemporaryDirectory() as tmp:
            local_input = os.path.join(tmp, os.path.basename(config.input_key))
            log(f"[{config.asset_id}] download s3://{config.bucket}/{config.input_key}")
            s3_client.download_file(config.bucket, config.input_key, local_input)

            pipeline_result = await _run_pipeline(
                local_input,
                target_lang=config.target_lang,
                source_lang=config.source_lang,
                subtitle_fmt=config.subtitle_fmt,
            )

            outputs = _build_output_keys(config.project_id, config.target_lang, config.asset_id, config.subtitle_fmt)
            workdir = pipeline_result["workdir"]
            meta = pipeline_result["meta"]
            duration = meta.get("orig_duration")

            uploaded = {}
            subtitle_key = outputs["subtitle"]
            s3_client.upload_file(pipeline_result["subtitle_path"], config.bucket, subtitle_key)
            uploaded["subtitle"] = subtitle_key

            video_path = pipeline_result["video_path"]
            if not video_path:
                raise RuntimeError("mux stage did not produce output video")
            s3_client.upload_file(video_path, config.bucket, outputs["video"])
            uploaded["video"] = outputs["video"]

            audio_path = pipeline_result.get("audio_path")
            if audio_path and os.path.exists(audio_path):
                s3_client.upload_file(audio_path, config.bucket, outputs["audio"])
                uploaded["audio"] = outputs["audio"]

            # 산출물 색인 정보를 프로젝트 규격에 맞춘 manifest.json으로 정리한다.
            manifest = {
                "assetId": config.asset_id,
                "projectId": config.project_id,
                "task": "dub",
                "input": {
                    "bucket": config.bucket,
                    "key": config.input_key,
                    "type": config.input_ext,
                },
                "output": {
                    "subtitle": {"bucket": config.bucket, "key": outputs["subtitle"], "format": config.subtitle_fmt},
                    "audio": {"bucket": config.bucket, "key": outputs["audio"], "format": "wav"} if "audio" in uploaded else None,
                    "video": {"bucket": config.bucket, "key": outputs["video"], "format": "mp4"},
                },
                "meta": {
                    "jobId": pipeline_result["job_id"],
                    "workdir": workdir,
                    "sourceLang": config.source_lang,
                    "targetLang": config.target_lang,
                    "segments": len(meta.get("segments", [])),
                    "translations": len(meta.get("translations", [])),
                    "duration": duration,
                },
                "createdAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            if manifest["output"]["audio"] is None:
                manifest["output"].pop("audio")

            manifest_path = os.path.join(tmp, "manifest.json")
            with open(manifest_path, "w", encoding="utf-8") as mf:
                json.dump(manifest, mf, ensure_ascii=False, indent=2)
            s3_client.upload_file(manifest_path, config.bucket, outputs["manifest"])
            uploaded["manifest"] = outputs["manifest"]

        return {"job_id": pipeline_result["job_id"], "uploaded": uploaded}


# --------------------------------------------------------------------------- #
# FastAPI lifecycle 연동용 헬퍼

_worker_instance: Optional[AWSLongPollingWorker] = None


def start_background_worker():
    global _worker_instance
    # 환경 변수로 워커 동작을 토글할 수 있게 한다.
    if not ENABLE_WORKER:
        return
    if _worker_instance and _worker_instance.is_running():
        return
    _worker_instance = AWSLongPollingWorker()
    _worker_instance.start()


def stop_background_worker():
    global _worker_instance
    if _worker_instance:
        _worker_instance.stop()
        _worker_instance = None


if __name__ == "__main__":
    worker = AWSLongPollingWorker()
    worker.start()
    try:
        while worker.is_running():
            time.sleep(1)
    except KeyboardInterrupt:
        log("KeyboardInterrupt received; stopping worker.")
    finally:
        worker.stop()
