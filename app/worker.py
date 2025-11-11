import asyncio
import json
import logging
import os
import time
from pathlib import Path

import boto3
import requests
from botocore.exceptions import ClientError, BotoCoreError

# 워커의 서비스 모듈들
from services.stt import run_asr
from services.translate import translate_transcript
from services.tts import generate_tts
from services.sync import sync_segments
from services.mux import mux_audio_video
from config import ensure_job_dirs

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
noisy_loggers = ['boto3', 'botocore', 's3transfer', 'urllib3']
for logger_name in noisy_loggers:
    logging.getLogger(logger_name).setLevel(logging.WARNING)

# AWS 설정
AWS_REGION = os.getenv("AWS_REGION", "ap-northeast-2")
JOB_QUEUE_URL = os.getenv("JOB_QUEUE_URL")
AWS_S3_BUCKET = os.getenv("AWS_S3_BUCKET")

if not all([JOB_QUEUE_URL, AWS_S3_BUCKET]):
    raise ValueError("JOB_QUEUE_URL and AWS_S3_BUCKET environment variables must be set.")

sqs_client = boto3.client("sqs", region_name=AWS_REGION)
s3_client = boto3.client("s3", region_name=AWS_REGION)


def send_callback(callback_url: str, status: str, message: str, stage: str | None = None, metadata: dict | None = None):
    """백엔드로 진행 상황 콜백을 보냅니다."""
    try:
        payload = {"status": status, "message": message}
        
        # 메타데이터 구성
        callback_metadata = metadata or {}
        if stage:
            callback_metadata["stage"] = stage
        
        if callback_metadata:
            payload["metadata"] = callback_metadata

        response = requests.post(callback_url, json=payload, timeout=10)
        response.raise_for_status()
        logging.info(f"Sent callback to {callback_url} with status: {status}, stage: {stage or 'N/A'}")
    except requests.RequestException as e:
        logging.error(f"Failed to send callback to {callback_url}: {e}")


def download_from_s3(bucket: str, key: str, local_path: Path) -> bool:
    """S3에서 파일을 다운로드합니다."""
    try:
        logging.info(f"Downloading s3://{bucket}/{key} to {local_path}...")
        local_path.parent.mkdir(parents=True, exist_ok=True)
        s3_client.download_file(bucket, key, str(local_path))
        logging.info(f"Successfully downloaded s3://{bucket}/{key}")
        return True
    except ClientError as e:
        logging.error(f"Failed to download from S3: {e}")
        return False


def upload_to_s3(bucket: str, key: str, local_path: Path) -> bool:
    """S3로 파일을 업로드합니다."""
    try:
        logging.info(f"Uploading {local_path} to s3://{bucket}/{key}...")
        s3_client.upload_file(str(local_path), bucket, key)
        logging.info(f"Successfully uploaded to s3://{bucket}/{key}")
        return True
    except ClientError as e:
        logging.error(f"Failed to upload to S3: {e}")
        return False
    except FileNotFoundError:
        logging.error(f"Local file not found for upload: {local_path}")
        return False


def full_pipeline(job_details: dict):
    """전체 더빙 파이프라인을 실행합니다."""
    job_id = job_details["job_id"]
    project_id = job_details["project_id"]
    input_key = job_details["input_key"]
    callback_url = job_details["callback_url"]
    
    target_lang = job_details.get("target_lang", "en") 
    voice_config = job_details.get("voice_config")

    send_callback(callback_url, "in_progress", f"Starting full pipeline for job {job_id}", stage="starting")

    # 1. 로컬 작업 디렉토리 설정
    paths = ensure_job_dirs(job_id)
    source_video_path = paths.input_dir / Path(input_key).name

    # 2. S3에서 원본 영상 다운로드
    if not download_from_s3(AWS_S3_BUCKET, input_key, source_video_path):
        send_callback(callback_url, "failed", "Failed to download source video from S3.", stage="download_failed")
        return

    # 3. voice_config에서 사용자 음성 샘플 다운로드 (필요 시)
    user_voice_sample_path = None
    if voice_config and voice_config.get("kind") == "s3" and voice_config.get("key"):
        voice_key = voice_config["key"]
        user_voice_sample_path = paths.interim_dir / Path(voice_key).name
        if not download_from_s3(AWS_S3_BUCKET, voice_key, user_voice_sample_path):
            send_callback(callback_url, "failed", f"Failed to download voice sample from S3 key: {voice_key}", stage="download_failed")
            return
        send_callback(callback_url, "in_progress", "Custom voice sample downloaded.", stage="downloaded")

    try:
        # 4. ASR (STT)
        send_callback(callback_url, "in_progress", "Starting ASR...", stage="asr_started")
        run_asr(job_id, source_video_path)
        # ASR 결과물(compact transcript)을 S3에 업로드
        from services.transcript_store import COMPACT_ARCHIVE_NAME
        asr_result_path = paths.src_sentence_dir / COMPACT_ARCHIVE_NAME
        upload_to_s3(AWS_S3_BUCKET, f"projects/{project_id}/interim/{job_id}/{COMPACT_ARCHIVE_NAME}", asr_result_path)
        send_callback(callback_url, "in_progress", "ASR completed.", stage="asr_completed")

        # 5. 번역
        send_callback(callback_url, "in_progress", "Starting translation...", stage="translation_started")
        translate_transcript(job_id, target_lang)
        # 번역 결과물(translated.json)을 S3에 업로드
        trans_result_path = paths.trg_sentence_dir / "translated.json"
        upload_to_s3(AWS_S3_BUCKET, f"projects/{project_id}/interim/{job_id}/translated.json", trans_result_path)
        send_callback(callback_url, "in_progress", "Translation completed.", stage="translation_completed")

        # 6. TTS
        send_callback(callback_url, "in_progress", "Starting TTS...", stage="tts_started")
        generate_tts(job_id, target_lang, voice_sample_path=user_voice_sample_path)
        # TTS 결과물(개별 wav 파일 및 segments.json)을 S3에 업로드
        tts_dir = paths.vid_tts_dir
        for tts_file in tts_dir.glob('**/*'):
            if tts_file.is_file():
                tts_key = f"projects/{project_id}/interim/{job_id}/tts/{tts_file.relative_to(tts_dir)}"
                upload_to_s3(AWS_S3_BUCKET, str(tts_key), tts_file)
        send_callback(callback_url, "in_progress", "TTS completed.", stage="tts_completed")

        # 7. Sync
        send_callback(callback_url, "in_progress", "Starting sync...", stage="sync_started")
        sync_segments(job_id)
        # Sync 결과물(synced 디렉토리)을 S3에 업로드
        synced_dir = paths.vid_tts_dir / "synced"
        for sync_file in synced_dir.glob('**/*'):
            if sync_file.is_file():
                sync_key = f"projects/{project_id}/interim/{job_id}/synced/{sync_file.relative_to(synced_dir)}"
                upload_to_s3(AWS_S3_BUCKET, str(sync_key), sync_file)
        send_callback(callback_url, "in_progress", "Sync completed.", stage="sync_completed")

        # 8. Mux
        send_callback(callback_url, "in_progress", "Starting mux...", stage="mux_started")
        mux_results = mux_audio_video(job_id, source_video_path)
        output_video_path = Path(mux_results["output_video"])
        
        # 9. 최종 결과물 S3에 업로드
        output_key = f"projects/{project_id}/outputs/{job_id}/{output_video_path.name}"
        if not upload_to_s3(AWS_S3_BUCKET, output_key, output_video_path):
             raise Exception("Failed to upload final video to S3")
        
        send_callback(callback_url, "done", "Pipeline completed successfully.", stage="done", metadata={"result_key": output_key})

    except Exception as e:
        logging.error(f"Pipeline failed for job {job_id}: {e}", exc_info=True)
        send_callback(callback_url, "failed", str(e), stage="failed")


async def poll_sqs():
    """SQS 큐를 폴링하여 메시지를 처리합니다."""
    logging.info("Starting SQS poller...")
    while True:
        try:
            response = sqs_client.receive_message(
                QueueUrl=JOB_QUEUE_URL,
                MaxNumberOfMessages=1,
                WaitTimeSeconds=20,
                MessageAttributeNames=['All']
            )

            messages = response.get("Messages", [])
            if not messages:
                continue

            for message in messages:
                receipt_handle = message["ReceiptHandle"]
                try:
                    logging.info(f"Received message: {message['MessageId']}")
                    job_details = json.loads(message["Body"])
                    
                    # 파이프라인 실행
                    full_pipeline(job_details)

                    # 처리 완료 후 메시지 삭제
                    sqs_client.delete_message(
                        QueueUrl=JOB_QUEUE_URL,
                        ReceiptHandle=receipt_handle
                    )
                    logging.info(f"Deleted message: {message['MessageId']}")

                except json.JSONDecodeError as e:
                    logging.error(f"Invalid JSON in message body: {e}")
                    # 잘못된 형식의 메시지는 큐에서 삭제
                    sqs_client.delete_message(QueueUrl=JOB_QUEUE_URL, ReceiptHandle=receipt_handle)
                except Exception as e:
                    logging.error(f"Error processing message {message.get('MessageId', 'N/A')}: {e}")
                    # 처리 중 에러 발생 시, 메시지를 바로 삭제하지 않고 SQS의 Visibility Timeout에 따라 재처리되도록 둡니다.
                    # 또는 Dead Letter Queue로 보내는 정책을 사용할 수 있습니다.
                    # 여기서는 일단 로깅만 하고 넘어갑니다.
                    pass

        except (BotoCoreError, ClientError) as e:
            logging.error(f"Error polling SQS: {e}")
            await asyncio.sleep(10) # 에러 발생 시 잠시 대기 후 재시도
        except Exception as e:
            logging.error(f"An unexpected error occurred in the poller: {e}")
            await asyncio.sleep(10)


if __name__ == "__main__":
    asyncio.run(poll_sqs())
