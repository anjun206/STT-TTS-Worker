import os
from app.config import JOB_QUEUE_URL, AWS_REGION, AWS_S3_BUCKET, DEFAULT_TARGET_LANG, DEFAULT_SOURCE_LANG, VISIBILITY_TIMEOUT, POLL_WAIT, CALLBACK_LOCALHOST_HOST, PROFILE, LOG_LEVEL
from botocore.exceptions import BotoCoreError, ClientError
import logging, time, requests, boto3, json
from app.config import JobProcessingError
from typing import Any, Dict, Optional
from .pipeline.full_pipeline import FullPipeline
from .pipeline.segment_tts import SegmentTTS
from .pipeline.segment_mix import SegmentMix
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)

class QueueWorker:
    def __init__(self):
        self.queue_url = JOB_QUEUE_URL
        self.bucket = AWS_S3_BUCKET
        self.default_target_lang = DEFAULT_TARGET_LANG
        self.default_source_lang = DEFAULT_SOURCE_LANG
        self.visibility_timeout = VISIBILITY_TIMEOUT
        self.poll_wait = POLL_WAIT
        self.callback_localhost_host = CALLBACK_LOCALHOST_HOST
        session_kwargs: dict = {}
        if PROFILE:
            session_kwargs["profile_name"] = PROFILE
        boto_session = boto3.Session(region_name=AWS_REGION, **session_kwargs)
        self.sqs_client = boto_session.client("sqs", region_name=AWS_REGION)
        self.s3_client = boto_session.client("s3", region_name=AWS_REGION)
        self.http = requests.Session()

    def poll_forever(self):
        while True:
            try:
                messages = self.sqs.receive_message(
                    QueueUrl=self.queue_url,
                    MaxNumberOfMessages=1,
                    WaitTimeSeconds=self.poll_wait,
                    VisibilityTimeout=self.visibility_timeout,
                    MessageAttributeNames=["All"],
                    AttributeNames=["All"],
                )
            except (BotoCoreError, ClientError) as exc:
                logger.error("Failed to poll SQS: %s", exc)
                time.sleep(5)
                continue

            for message in messages.get("Messages", []):
                receipt = message["ReceiptHandle"]
                rc = int(message.get("Attributes", {}).get("ApproximateReceiveCount", "1"))
                logger.info("SQS received (ReceiveCount=%s, MessageId=%s)", rc, message.get("MessageId"))

                try:
                    body = json.loads(message.get("Body", "{}"))
                except json.JSONDecodeError:
                    logger.error("Invalid message body, deleting: %s", message.get("Body"))
                    self._delete_message(receipt)
                    continue

                success = False
                try:
                    self.__handle_job(body)
                    success = True
                except JobProcessingError as exc:
                    logger.error("Job %s failed: %s", body.get("job_id"), exc)
                except Exception as exc:  # pylint: disable=broad-except
                    logger.exception("Unexpected error handling message: %s", exc)

                finally:
                    self.sqs.delete_message(QueueUrl=self.queue_url, ReceiptHandle=receipt)
                    logger.info("SQS deleted: %s", message.get("MessageId"))

    def __handle_job(self, payload: Dict[str, Any]) -> None:
        task = (payload.get("task") or "full_pipeline").lower()

        if task == "full_pipeline":  # preTTS
            full_pipeline = FullPipeline(payload=payload)
            full_pipeline.process()
            # 최종 TTS + mix -> 최종 결과물 나오는 ~ handle_최종 어쩌구 저쩌구(payload) 필요
        else:
            raise JobProcessingError(f"Unsupported task type: {task}")

if __name__ == "__main__":
    worker = QueueWorker()
    worker.poll_forever()
