from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from botocore.exceptions import BotoCoreError, ClientError

from app.config import (
    AWS_S3_BUCKET,
    DEFAULT_SOURCE_LANG,
    DEFAULT_TARGET_LANG,
    JobProcessingError,
    LOG_LEVEL,
    ensure_job_dirs,
    post_status,
)
from app.services.mux import mux_audio_video
from app.services.stt import run_asr
from app.services.sync import sync_segments
from app.services.translate import translate_transcript
from app.services.tts import generate_tts


logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )
else:
    logger.setLevel(LOG_LEVEL)


class FullPipeline:
    """Executes the end-to-end dubbing pipeline for a single SQS job."""

    def __init__(
        self,
        payload: Dict[str, Any],
        *,
        s3_client,
        http,
        input_bucket: Optional[str] = None,
        output_bucket: Optional[str] = None,
    ):
        self.payload = payload
        self.s3_client = s3_client
        self.http = http

        self.bucket = payload.get("input_bucket") or input_bucket or AWS_S3_BUCKET
        self.output_bucket = (
            payload.get("output_bucket") or output_bucket or self.bucket
        )

        self.job_id = (payload.get("job_id") or "").strip()
        self.project_id = payload.get("project_id")
        self.input_key = (payload.get("input_key") or "").strip()
        self.callback_url = (payload.get("callback_url") or "").strip()
        self.target_lang = (
            payload.get("target_lang") or DEFAULT_TARGET_LANG
        ).strip()
        self.source_lang = (
            payload.get("source_lang") or DEFAULT_SOURCE_LANG
        ).strip()
        self.voice_sample_key = payload.get("voice_sample_key")
        self.voice_sample_bucket = payload.get("voice_sample_bucket") or self.bucket
        self.voice_sample_path_hint = payload.get("voice_sample_path")
        raw_prompt = payload.get("prompt_text") or payload.get("prompt_text_value")
        self.prompt_text = raw_prompt.strip() if isinstance(raw_prompt, str) else None

        self.output_prefix = self._resolve_output_prefix(payload.get("output_prefix"))
        self.result_key = payload.get("result_key") or f"{self.output_prefix}/videos/{self.job_id}.mp4"
        self.metadata_key = payload.get("metadata_key") or f"{self.output_prefix}/metadata/{self.job_id}.json"

        self.paths = ensure_job_dirs(self.job_id) if self.job_id else None
        self.local_input = None

    def process(self) -> Dict[str, Any]:
        self._validate_payload()
        try:
            # 1) 잡 승인 및 입력 다운로드
            self._post_stage("accepted", {"job_id": self.job_id})

            self.local_input = self.paths.input_dir / "source.mp4"
            self._download_source()
            self._post_stage(
                "downloaded", {"input_key": self.input_key, "bucket": self.bucket}
            )

            # 2) ASR → 번역 → TTS → 싱크 순서로 미디어를 준비한다.
            run_asr(self.job_id)
            self._post_stage("stt_completed")

            translations = translate_transcript(self.job_id, self.target_lang)
            self._post_stage(
                "translation_completed", {"segments_translated": len(translations)}
            )

            voice_sample_path = self._prepare_voice_sample()
            segments_payload = generate_tts(
                self.job_id,
                self.target_lang,
                voice_sample_path=voice_sample_path,
                prompt_text_override=self.prompt_text,
            )
            self._post_stage("tts_completed", {"segments": len(segments_payload)})

            try:
                synced_segments = sync_segments(self.job_id)
            except FileNotFoundError as exc:
                logger.info("싱크 입력이 없어 건너뜁니다: %s", exc)
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning("싱크 단계가 실패했습니다. 기존 세그먼트를 그대로 사용합니다: %s", exc)
            else:
                if synced_segments:
                    segments_payload = synced_segments
                    self._post_stage("sync_completed", {"segments": len(segments_payload)})

            # 3) 믹싱 및 결과 업로드
            mux_outputs = mux_audio_video(self.job_id)
            result_video_path = Path(mux_outputs["output_video"])
            final_audio_path = Path(mux_outputs["output_audio"])
            self._post_stage("mux_completed", {"result_video": str(result_video_path)})

            self._upload_file(result_video_path, self.output_bucket, self.result_key)
            metadata_payload = self._build_metadata(
                segments_payload, translations, final_audio_path
            )
            self._upload_metadata(metadata_payload)
            self._post_stage(
                "upload_completed",
                {
                    "result_bucket": self.output_bucket,
                    "result_key": self.result_key,
                    "metadata_key": self.metadata_key,
                },
            )

            return {
                "job_id": self.job_id,
                "project_id": self.project_id,
                "result_bucket": self.output_bucket,
                "result_key": self.result_key,
                "metadata_key": self.metadata_key,
                "segments": segments_payload,
                "segment_count": len(segments_payload),
                "target_lang": self.target_lang,
                "source_lang": self.source_lang,
            }
        except JobProcessingError:
            raise
        except (BotoCoreError, ClientError) as exc:
            raise JobProcessingError(f"AWS 클라이언트 오류가 발생했습니다: {exc}") from exc
        except Exception as exc:  # pylint: disable=broad-except
            raise JobProcessingError(str(exc)) from exc

    # Helpers -----------------------------------------------------------------

    def _validate_payload(self) -> None:
        if not self.job_id:
            raise JobProcessingError("payload 에 job_id 가 없습니다")
        if not self.input_key:
            raise JobProcessingError("payload 에 input_key 가 없습니다")
        if not self.callback_url:
            raise JobProcessingError("payload 에 callback_url 이 없습니다")
        if self.paths is None:
            self.paths = ensure_job_dirs(self.job_id)

    def _resolve_output_prefix(self, prefix: Optional[str]) -> str:
        if prefix:
            return prefix.rstrip("/")
        if self.project_id:
            return f"projects/{self.project_id}/{self.job_id}"
        return f"jobs/{self.job_id}"

    def _post_stage(self, stage: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        if not self.callback_url:
            return
        payload = {"stage": stage, "job_id": self.job_id}
        if metadata:
            payload.update(metadata)
        post_status(
            self.http,
            self.callback_url,
            "in_progress",
            stage_id=stage,
            metadata=payload,
            project_id=self.project_id,
        )

    def _download_source(self) -> None:
        assert self.local_input is not None
        self.local_input.parent.mkdir(parents=True, exist_ok=True)
        try:
            logger.info(
                "입력 영상을 내려받습니다 s3://%s/%s -> %s",
                self.bucket,
                self.input_key,
                self.local_input,
            )
            self.s3_client.download_file(
                self.bucket, self.input_key, str(self.local_input)
            )
        except (BotoCoreError, ClientError) as exc:
            raise JobProcessingError(f"입력 다운로드에 실패했습니다: {exc}") from exc

    def _prepare_voice_sample(self) -> Optional[Path]:
        if isinstance(self.voice_sample_path_hint, str):
            candidate = Path(self.voice_sample_path_hint).expanduser()
            if candidate.is_file():
                return candidate
        if not self.voice_sample_key:
            return None
        ref_dir = self.paths.interim_dir / "tts_custom_refs"
        ref_dir.mkdir(parents=True, exist_ok=True)
        suffix = Path(self.voice_sample_key).suffix or ".wav"
        dest = ref_dir / f"user_voice_sample{suffix}"
        try:
            logger.info(
                "보이스 샘플을 내려받습니다 s3://%s/%s -> %s",
                self.voice_sample_bucket,
                self.voice_sample_key,
                dest,
            )
            self.s3_client.download_file(
                self.voice_sample_bucket, self.voice_sample_key, str(dest)
            )
        except (BotoCoreError, ClientError) as exc:
            raise JobProcessingError(f"보이스 샘플 다운로드에 실패했습니다: {exc}") from exc
        return dest

    def _upload_file(self, path: Path, bucket: str, key: str) -> None:
        try:
            logger.info("%s 을(를) s3://%s/%s 로 업로드합니다", path, bucket, key)
            self.s3_client.upload_file(str(path), bucket, key)
        except (BotoCoreError, ClientError) as exc:
            raise JobProcessingError(f"결과 업로드에 실패했습니다: {exc}") from exc

    def _upload_metadata(self, metadata: Dict[str, Any]) -> None:
        body = json.dumps(metadata, ensure_ascii=False).encode("utf-8")
        try:
            self.s3_client.put_object(
                Bucket=self.output_bucket,
                Key=self.metadata_key,
                Body=body,
                ContentType="application/json",
            )
        except (BotoCoreError, ClientError) as exc:
            raise JobProcessingError(f"메타데이터 업로드에 실패했습니다: {exc}") from exc

    def _build_metadata(
        self,
        segments: list[Dict[str, Any]],
        translations: list[Dict[str, Any]],
        audio_path: Path,
    ) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "project_id": self.project_id,
            "target_lang": self.target_lang,
            "source_lang": self.source_lang,
            "input_bucket": self.bucket,
            "input_key": self.input_key,
            "result_bucket": self.output_bucket,
            "result_key": self.result_key,
            "metadata_key": self.metadata_key,
            "segments": segments,
            "segment_count": len(segments),
            "translations": translations,
            "audio_artifact": str(audio_path),
        }
