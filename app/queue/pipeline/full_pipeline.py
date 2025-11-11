from typing import Any, Dict
from app.config import JobProcessingError
import os, logging
from app.config.env import DEFAULT_TARGET_LANG, DEFAULT_SOURCE_LANG, LOG_LEVEL
from app.config.utils import post_status, ensure_workdir
from app.services.stt import run_asr
from app.services.translate import translate_transcript
from app.services.tts import generate_tts
from app.services.mux import mux_audio_video
from app.services.sync import sync_segments
import json
from botocore.exceptions import BotoCoreError, ClientError


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)

class FullPipeline:
    def __init__(self, payload: Dict[str, Any]):
        self.job_id = payload.get("job_id")
        self.project_id = payload.get("project_id")
        self.input_key = payload.get("input_key")
        self.callback_url = payload.get("callback_url")
        self.__validation_check()
        self.target_lang = payload.get("target_lang") or DEFAULT_TARGET_LANG
        self.source_lang = payload.get("source_lang") or DEFAULT_SOURCE_LANG
        self.workdir = ensure_workdir(self.job_id)
        self.extension = os.path.splitext(self.input_key)[1]
        self.local_input = os.path.join(self.workdir, f"input{self.extension or '.mp4'}")
        self.user_voice_sample_path = payload.get("user_voice_sample_path")
        self.prompt_text_value = payload.get("prompt_text_value")
        self.prompt_text_value = self.prompt_text.strip() if self.prompt_text else None

    def process(self):
        try:
            logger.info("Downloading s3://%s/%s to %s", self.bucket, self.input_key, self.local_input)
            self.s3_client.download_file(self.bucket, self.input_key, self.local_input)

            post_status(
                self.self.callback_url, 
                "in_progress", 
                stage_status="processing",
                project_id=self.project_id,
                metadata={
                    "stage": "downloaded",
                    "job_id": self.job_id,
                    "project_id": self.project_id,
                },
            )

            meta = run_asr(self.job_id, self.local_input)

            post_status(self.callback_url, "in_progress", metadata={"stage": "stt_completed"})
            post_status(self.callback_url, "in_progress", metadata={"stage": "mt_prepare"})

            translations = translate_transcript(self.job_id, self.target_lang)
            

            meta["translations"] = translations
            # save_meta(workdir, meta)

            post_status(self.callback_url, "in_progress", metadata={"stage": "mt_completed"})
            post_status(
                self.callback_url,
                "in_progress",
                stage_id="tts",
                stage_status="processing",
                project_id=self.project_id,
                metadata={
                    "stage": "tts_prepare",
                    "job_id": self.job_id,
                    "project_id": self.project_id,
                },
            )

            segments_payload = generate_tts(
                self.job_id,
                self.target_lang,
                voice_sample_path=self.user_voice_sample_path,
                prompt_text_override=self.prompt_text_value,
            )
            try:
                synced_segments = sync_segments(self.job_id)
            except FileNotFoundError as exc:
                logger.info("Skipping sync step due to missing inputs: %s", exc)
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning("Sync step failed; continuing without synced segments: %s", exc)
            else:
                if synced_segments:
                    segments_payload = synced_segments

            result_key = f"projects/{self.project_id}/outputs/videos/{self.job_id}.mp4"
            metadata_key = f"projects/{self.project_id}/outputs/metadata/{self.job_id}.json"

            logger.info("Uploading result video to s3://%s/%s", self.bucket, result_key)

            output_path = mux_audio_video(self.job_id)
            self.s3_client.upload_file(output_path, self.bucket, result_key)
            self.s3_client.put_object(
                Bucket=self.bucket,
                Key=metadata_key,
                Body=json.dumps(
                    {
                        "job_id": self.job_id,
                        "project_id": self.project_id,
                        "segments": segments_payload,
                        "target_lang": self.target_lang,
                        "source_lang": self.source_lang,
                        "input_key": self.input_key,
                        "result_key": result_key,
                    },
                    ensure_ascii=False,
                ).encode("utf-8"),
                ContentType="application/json",
            )

            status_payload = {
                "stage": "tts_completed",
                "segments_count": len(segments_payload),
                "segments": segments_payload,
                "metadata_key": metadata_key,
                "result_key": result_key,
                "target_lang": self.target_lang,
                "source_lang": self.source_lang,
                "input_key": self.input_key,
                "segment_assets_prefix": f"projects/{self.project_id}/interim/{self.job_id}/segments"
            }
            post_status(
                self.callback_url,
                "done",
                stage_id="mux",
                stage_status="done",
                project_id=self.project_id,
                result_key=result_key,
                metadata=status_payload,
            )
        except (BotoCoreError, ClientError) as exc:
            failure = JobProcessingError(f"AWS client error: {exc}")
            self._safe_fail(self.callback_url, failure)
            raise failure
        except JobProcessingError as exc:
            self._safe_fail(self.callback_url, exc)
            raise
        except Exception as exc:  # pylint: disable=broad-except
            wrapped = JobProcessingError(str(exc))
            self._safe_fail(self.callback_url, wrapped)
            raise wrapped
    
    def __validation_check(self):
        if not all([self.job_id, self.project_id, self.input_key, self.self.callback_url]):
            raise JobProcessingError("Missing required job fields in payload")
