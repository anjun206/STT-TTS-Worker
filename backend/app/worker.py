import asyncio
import json
import logging
import os
import shlex
import shutil
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse, urlunparse

import boto3
import requests
from botocore.exceptions import BotoCoreError, ClientError

from .pipeline import (
    TTS_MODEL,
    _annotate_segments,
    _extract_tracks,
    _whisper_transcribe,
    mux_stage,
    translate_stage,
    tts_finalize_stage,
)
from .utils import (
    cut_wav_segment,
    concat_audio,
    ffprobe_duration,
    mask_keep_intervals,
    mix_bgm_with_tts,
    run,
    trim_or_pad_to_duration,
    detect_mean_volume_db,
    apply_gain_db,
)
from .tts import synthesize
from .utils_meta import load_meta, save_meta
from .vad import (
    complement_intervals,
    compute_vad_silences,
    merge_intervals,
    sum_silence_between,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=os.getenv("WORKER_LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)


class JobProcessingError(Exception):
    """Raised when a job fails irrecoverably during processing."""


def _ensure_workdir(job_id: str) -> str:
    workdir = os.path.join("/app/data", job_id)
    os.makedirs(workdir, exist_ok=True)
    return workdir


def _run_asr_stage(job_id: str, input_path: str) -> Dict[str, Any]:
    """
    Execute ASR pipeline for a given job and persist metadata.
    """
    workdir = _ensure_workdir(job_id)
    full_48k, vocals_48k, bgm_48k, vocals_16k_raw = _extract_tracks(input_path, workdir)
    total = ffprobe_duration(full_48k)

    silences = compute_vad_silences(
        vocals_16k_raw,
        aggressiveness=int(os.getenv("VAD_AGGR", "3")),
        frame_ms=int(os.getenv("VAD_FRAME_MS", "30")),
    )

    segments = _whisper_transcribe(vocals_16k_raw)

    margin = float(os.getenv("STT_INTERVAL_MARGIN", "0.10"))
    stt_intervals = merge_intervals(
        [
            (
                max(0.0, float(s["start"]) - margin),
                min(float(total), float(s["end"]) + margin),
            )
            for s in segments
            if float(s["end"]) > float(s["start"])
        ]
    )

    speech_only_48k = os.path.join(workdir, "speech_only_48k.wav")
    vocals_fx_48k = os.path.join(workdir, "vocals_fx_48k.wav")
    mask_keep_intervals(vocals_48k, stt_intervals, speech_only_48k, sr=48000, ac=2)
    nonspeech_intervals = complement_intervals(stt_intervals, total)
    mask_keep_intervals(vocals_48k, nonspeech_intervals, vocals_fx_48k, sr=48000, ac=2)

    wav_16k = os.path.join(workdir, "speech_16k.wav")
    run(
        f"ffmpeg -y -i {shlex.quote(speech_only_48k)} -ac 1 -ar 16000 -c:a pcm_s16le {shlex.quote(wav_16k)}"
    )

    for i in range(len(segments)):
        if i < len(segments) - 1:
            st = float(segments[i]["end"])
            en = float(segments[i + 1]["start"])
            segments[i]["gap_after_vad"] = sum_silence_between(silences, st, en)
            segments[i]["gap_after"] = max(0.0, en - st)
        else:
            segments[i]["gap_after_vad"] = 0.0
            segments[i]["gap_after"] = 0.0

    _annotate_segments(segments)

    meta = {
        "job_id": job_id,
        "workdir": workdir,
        "input": input_path,
        "audio_full_48k": full_48k,
        "vocals_48k": vocals_48k,
        "bgm_48k": bgm_48k,
        "speech_only_48k": speech_only_48k,
        "vocals_fx_48k": vocals_fx_48k,
        "wav_16k": wav_16k,
        "orig_duration": total,
        "segments": segments,
        "silences": silences,
        "speech_intervals_stt": stt_intervals,
        "nonspeech_intervals_stt": nonspeech_intervals,
    }
    save_meta(workdir, meta)
    return meta


def _build_segment_payload(
    meta: Dict[str, Any], translations: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    payload: List[Dict[str, Any]] = []
    segs = meta.get("segments") or []
    for idx, seg in enumerate(segs):
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", 0.0))
        length = seg.get("length")
        if length is None:
            length = max(0.0, end - start)
        translation_text = ""
        if idx < len(translations):
            translation_text = translations[idx].get("text", "")
        assets = seg.get("assets") or {}
        payload.append(
            {
                "segment_id": str(seg.get("seg_id", idx)),
                "segment_text": seg.get("text", ""),
                "score": seg.get("score"),
                "start_point": start,
                "end_point": end,
                "editor_id": None,
                "translate_context": translation_text,
                "sub_langth": length,
                "issues": seg.get("issues", []),
                # legacy keys (for compatibility with older consumers)
                "seg_id": seg.get("seg_id", idx),
                "seg_txt": seg.get("text", ""),
                "start": start,
                "end": end,
                "length": length,
                "editor": None,
                "trans_txt": translation_text,
                "assets": assets,
                "source_key": seg.get("source_key") or assets.get("source_key"),
                "bgm_key": seg.get("bgm_key") or assets.get("bgm_key"),
                "tts_key": seg.get("tts_key") or assets.get("tts_key"),
                "mix_key": seg.get("mix_key") or assets.get("mix_key"),
                "video_key": seg.get("video_key") or assets.get("video_key"),
            }
        )
    return payload


class QueueWorker:
    def __init__(self) -> None:
        self.queue_url = os.environ["JOB_QUEUE_URL"]
        self.bucket = os.environ["AWS_S3_BUCKET"]
        self.region = os.getenv("AWS_REGION", "ap-northeast-2")
        self.default_target_lang = os.getenv("JOB_TARGET_LANG", "en")
        self.default_source_lang = os.getenv("JOB_SOURCE_LANG", "ko")
        self.result_video_prefix = os.getenv(
            "JOB_RESULT_VIDEO_PREFIX",
            "projects/{project_id}/outputs/videos/{job_id}.mp4",
        )
        self.result_meta_prefix = os.getenv(
            "JOB_RESULT_METADATA_PREFIX",
            "projects/{project_id}/outputs/metadata/{job_id}.json",
        )
        self.interim_segment_prefix = os.getenv(
            "JOB_INTERIM_SEGMENT_PREFIX",
            "projects/{project_id}/interim/{job_id}/segments",
        )
        self.visibility_timeout = int(os.getenv("JOB_VISIBILITY_TIMEOUT", "300"))
        self.poll_wait = int(os.getenv("JOB_QUEUE_WAIT", "20"))
        self.callback_localhost_host = os.getenv(
            "JOB_CALLBACK_LOCALHOST_HOST", "host.docker.internal"
        )

        session_kwargs: Dict[str, Any] = {}
        profile = os.getenv("AWS_PROFILE")
        if profile:
            session_kwargs["profile_name"] = profile
        self.boto_session = boto3.Session(region_name=self.region, **session_kwargs)
        self.sqs = self.boto_session.client("sqs", region_name=self.region)
        self.s3 = self.boto_session.client("s3", region_name=self.region)
        self.http = requests.Session()

    def poll_forever(self) -> None:
        logger.info("Starting SQS poller for queue %s", self.queue_url)
        while True:
            try:
                messages = self.sqs.receive_message(
                    QueueUrl=self.queue_url,
                    MaxNumberOfMessages=1,
                    WaitTimeSeconds=self.poll_wait,
                    VisibilityTimeout=self.visibility_timeout,
                    MessageAttributeNames=["All"],
                )
            except (BotoCoreError, ClientError) as exc:
                logger.error("Failed to poll SQS: %s", exc)
                time.sleep(5)
                continue

            for msg in messages.get("Messages", []):
                receipt = msg["ReceiptHandle"]
                try:
                    body = json.loads(msg.get("Body", "{}"))
                except json.JSONDecodeError:
                    logger.error("Invalid message body, deleting: %s", msg.get("Body"))
                    self._delete_message(receipt)
                    continue

                success = False
                try:
                    self._handle_job(body)
                    success = True
                except JobProcessingError as exc:
                    logger.error("Job %s failed: %s", body.get("job_id"), exc)
                except Exception as exc:  # pylint: disable=broad-except
                    logger.exception("Unexpected error handling message: %s", exc)

                if success:
                    self._delete_message(receipt)

    def _delete_message(self, receipt: str) -> None:
        try:
            self.sqs.delete_message(QueueUrl=self.queue_url, ReceiptHandle=receipt)
        except (BotoCoreError, ClientError) as exc:
            logger.error("Failed to delete SQS message: %s", exc)

    def _handle_job(self, payload: Dict[str, Any]) -> None:
        task = (payload.get("task") or "full_pipeline").lower()

        if task == "full_pipeline":
            self._handle_full_pipeline(payload)
        elif task == "segment_tts":
            self._handle_segment_tts(payload)
        elif task in {"segment_mix", "tts_bgm_mix"}:
            self._handle_segment_mix(payload)
        else:
            raise JobProcessingError(f"Unsupported task type: {task}")

    def _upload_file(
        self, local_path: str, key: str, content_type: Optional[str] = None
    ) -> None:
        extra: Dict[str, Any] | None = None
        if content_type:
            extra = {"ContentType": content_type}
        kwargs: Dict[str, Any] = {}
        if extra:
            kwargs["ExtraArgs"] = extra
        self.s3.upload_file(local_path, self.bucket, key, **kwargs)

    def _prepare_segment_assets(
        self,
        project_id: str,
        job_id: str,
        meta: Dict[str, Any],
    ) -> None:
        segments = meta.get("segments") or []
        if not segments:
            return

        workdir = meta.get("workdir") or _ensure_workdir(job_id)
        segment_dir = os.path.join(workdir, "segments")
        os.makedirs(segment_dir, exist_ok=True)

        prefix = self.interim_segment_prefix.format(
            project_id=project_id, job_id=job_id
        ).rstrip("/")

        speech_src = meta.get("speech_only_48k") or meta.get("audio_full_48k")
        bgm_src = meta.get("bgm_48k")
        tts_src = meta.get("dubbed_wav")
        final_mix_src = meta.get("final_mix")
        mix_src = final_mix_src or meta.get("dubbed_wav")
        video_src = meta.get("input")

        def _cut_if_possible(
            src: Optional[str],
            dst: str,
            start: float,
            end: float,
            *,
            sample_rate: Optional[int] = None,
            channels: Optional[int] = None,
        ) -> bool:
            if not src or not os.path.exists(src):
                return False
            cmd = [
                "ffmpeg",
                "-y",
                "-ss",
                f"{start:.6f}",
                "-to",
                f"{end:.6f}",
                "-i",
                shlex.quote(src),
            ]
            if sample_rate:
                cmd.extend(["-ar", str(sample_rate)])
            if channels:
                cmd.extend(["-ac", str(channels)])
            if sample_rate or channels:
                cmd.extend(["-c:a", "pcm_s16le"])
            else:
                cmd.extend(["-c", "copy"])
            cmd.append(shlex.quote(dst))
            run(" ".join(cmd))
            return True

        for idx, seg in enumerate(segments):
            try:
                start = float(seg.get("start", 0.0))
                end = float(seg.get("end", 0.0))
            except (TypeError, ValueError):
                continue
            if end <= start + 1e-3:
                continue

            base_name = f"{idx:04d}"
            assets: Dict[str, Any] = {}

            # 원본 발화
            local_source = os.path.join(segment_dir, f"{base_name}_source.wav")
            has_source = _cut_if_possible(
                speech_src, local_source, start, end, sample_rate=16000, channels=1
            )
            if has_source:
                source_key = f"{prefix}/{base_name}_source.wav"
                self._upload_file(local_source, source_key, "audio/wav")
                assets["source_key"] = source_key

            # BGM
            local_bgm = os.path.join(segment_dir, f"{base_name}_bgm.wav")
            has_bgm = _cut_if_possible(
                bgm_src, local_bgm, start, end, sample_rate=48000, channels=2
            )
            if has_bgm:
                bgm_key = f"{prefix}/{base_name}_bgm.wav"
                self._upload_file(local_bgm, bgm_key, "audio/wav")
                assets["bgm_key"] = bgm_key

            # 합성 음성
            local_tts = os.path.join(segment_dir, f"{base_name}_tts.wav")
            has_tts = _cut_if_possible(
                tts_src, local_tts, start, end, sample_rate=48000, channels=1
            )
            if has_tts:
                tts_key = f"{prefix}/{base_name}_tts.wav"
                self._upload_file(local_tts, tts_key, "audio/wav")
                assets["tts_key"] = tts_key

            # 배경음이 섞인 최종 믹스(없으면 TTS로 폴백)
            local_mix = os.path.join(segment_dir, f"{base_name}_mix.wav")
            mix_created = False
            if final_mix_src and _cut_if_possible(
                final_mix_src, local_mix, start, end, sample_rate=48000, channels=2
            ):
                mix_created = True
            elif (
                has_bgm
                and has_tts
                and os.path.exists(local_bgm)
                and os.path.exists(local_tts)
            ):
                try:
                    mix_bgm_with_tts(local_bgm, local_tts, local_mix)
                    mix_created = True
                except Exception as exc:  # pragma: no cover
                    logger.warning(
                        "Failed to create bgm mix for segment %s: %s", base_name, exc
                    )
            elif mix_src and _cut_if_possible(
                mix_src, local_mix, start, end, sample_rate=48000, channels=2
            ):
                mix_created = True

            if mix_created and os.path.exists(local_mix):
                mix_key = f"{prefix}/{base_name}_mix.wav"
                self._upload_file(local_mix, mix_key, "audio/wav")
                assets["mix_key"] = mix_key

            # 구간 영상 (원본 오디오 포함)
            if video_src and os.path.exists(video_src):
                local_video = os.path.join(segment_dir, f"{base_name}_video.mp4")
                duration = max(0.05, end - start)
                run(
                    " ".join(
                        [
                            "ffmpeg -y",
                            f"-ss {start:.6f}",
                            f"-i {shlex.quote(video_src)}",
                            f"-t {duration:.6f}",
                            "-map 0:v:0",
                            "-map 0:a:0?",
                            "-c:v copy",
                            "-c:a copy",
                            "-movflags +faststart",
                            shlex.quote(local_video),
                        ]
                    )
                )
                video_key = f"{prefix}/{base_name}_video.mp4"
                self._upload_file(local_video, video_key, "video/mp4")
                assets["video_key"] = video_key

            if assets:
                seg.setdefault("assets", {}).update(assets)
                for key, value in assets.items():
                    seg[key] = value

        save_meta(workdir, meta)

    def _handle_full_pipeline(self, payload: Dict[str, Any]) -> None:
        job_id = payload.get("job_id")
        project_id = payload.get("project_id")
        input_key = payload.get("input_key")
        callback_url = payload.get("callback_url")
        if not all([job_id, project_id, input_key, callback_url]):
            raise JobProcessingError("Missing required job fields in payload")
        target_lang = payload.get("target_lang") or self.default_target_lang
        source_lang = payload.get("source_lang") or self.default_source_lang
        workdir = _ensure_workdir(job_id)
        extension = os.path.splitext(input_key)[1]
        local_input = os.path.join(workdir, f"input{extension or '.mp4'}")
        try:
            logger.info(
                "Downloading s3://%s/%s to %s", self.bucket, input_key, local_input
            )
            self.s3.download_file(self.bucket, input_key, local_input)
            self._post_status(
                callback_url, "in_progress", metadata={"stage": "downloaded"}
            )
            meta = _run_asr_stage(job_id, local_input)
            translations = translate_stage(
                meta["segments"], src=source_lang, tgt=target_lang
            )
            meta["translations"] = translations
            save_meta(workdir, meta)
            self._post_status(
                callback_url, "in_progress", metadata={"stage": "tts_prepare"}
            )
            asyncio.run(tts_finalize_stage(job_id, target_lang, None))
            output_path = mux_stage(job_id)
            meta = load_meta(workdir)
            self._prepare_segment_assets(project_id, job_id, meta)
            meta = load_meta(workdir)
            segments_payload = _build_segment_payload(meta, translations)
            result_key = self.result_video_prefix.format(
                project_id=project_id, job_id=job_id
            )
            metadata_key = self.result_meta_prefix.format(
                project_id=project_id, job_id=job_id
            )
            logger.info("Uploading result video to s3://%s/%s", self.bucket, result_key)
            self.s3.upload_file(output_path, self.bucket, result_key)
            self.s3.put_object(
                Bucket=self.bucket,
                Key=metadata_key,
                Body=json.dumps(
                    {
                        "job_id": job_id,
                        "project_id": project_id,
                        "segments": segments_payload,
                        "target_lang": target_lang,
                        "source_lang": source_lang,
                        "input_key": input_key,
                        "result_key": result_key,
                    },
                    ensure_ascii=False,
                ).encode("utf-8"),
                ContentType="application/json",
            )
            status_payload = {
                "stage": "completed",
                "segments_count": len(segments_payload),
                "segments": segments_payload,
                "metadata_key": metadata_key,
                "result_key": result_key,
                "target_lang": target_lang,
                "source_lang": source_lang,
                "input_key": input_key,
                "segment_assets_prefix": self.interim_segment_prefix.format(
                    project_id=project_id, job_id=job_id
                ),
            }
            self._post_status(
                callback_url,
                "done",
                result_key=result_key,
                metadata=status_payload,
            )
        except (BotoCoreError, ClientError) as exc:
            failure = JobProcessingError(f"AWS client error: {exc}")
            self._safe_fail(callback_url, failure)
            raise failure
        except JobProcessingError as exc:
            self._safe_fail(callback_url, exc)
            raise
        except Exception as exc:  # pylint: disable=broad-except
            wrapped = JobProcessingError(str(exc))
            self._safe_fail(callback_url, wrapped)
            raise wrapped

    def _handle_segment_tts(self, payload: Dict[str, Any]) -> None:
        job_id = payload.get("job_id")
        callback_url = payload.get("callback_url")
        segment_payload: Dict[str, Any] = payload.get("segment") or {}

        if not job_id or not callback_url:
            raise JobProcessingError("segment_tts requires job_id and callback_url")

        try:
            segment_index = int(segment_payload.get("segment_index"))
        except (TypeError, ValueError):
            raise JobProcessingError("segment_tts requires a numeric segment_index")

        text = segment_payload.get("text")
        if not text:
            raise JobProcessingError("segment_tts requires translated text")

        project_id = segment_payload.get("project_id") or payload.get("project_id")
        segment_id = segment_payload.get("segment_id")
        assets: Dict[str, Any] = segment_payload.get("assets") or {}
        source_keys: List[str] = [
            str(k) for k in segment_payload.get("source_keys") or [] if k
        ]
        assets_prefix = segment_payload.get("segment_assets_prefix") or payload.get(
            "segment_assets_prefix"
        )

        base_name = f"{segment_index:04d}"
        if not source_keys and assets.get("source_key"):
            source_keys.append(str(assets["source_key"]))
        if not source_keys and assets_prefix:
            source_keys.append(f"{assets_prefix.rstrip('/')}/{base_name}_source.wav")

        fallback_ref_key = assets.get("tts_key")
        workdir = os.path.join(_ensure_workdir(job_id), "segment_tts")
        os.makedirs(workdir, exist_ok=True)

        try:
            self._post_status(
                callback_url,
                "in_progress",
                metadata={
                    "stage": "segment_tts_started",
                    "job_id": job_id,
                    "project_id": project_id,
                    "segment_index": segment_index,
                },
            )

            local_refs: List[str] = []
            for idx, key in enumerate(source_keys):
                local_path = os.path.join(workdir, f"ref_{idx:02d}.wav")
                try:
                    self.s3.download_file(self.bucket, key, local_path)
                    local_refs.append(local_path)
                except (BotoCoreError, ClientError) as exc:
                    logger.warning("Failed to download source key %s: %s", key, exc)

            if not local_refs and fallback_ref_key:
                local_fallback = os.path.join(workdir, "ref_fallback.wav")
                try:
                    self.s3.download_file(self.bucket, fallback_ref_key, local_fallback)
                    local_refs.append(local_fallback)
                except (BotoCoreError, ClientError) as exc:
                    logger.warning(
                        "Failed to download fallback TTS key %s: %s",
                        fallback_ref_key,
                        exc,
                    )

            if not local_refs:
                raise JobProcessingError("No reference audio available for segment_tts")

            converted_refs: List[str] = []
            for path in local_refs:
                base = os.path.splitext(os.path.basename(path))[0]
                converted = os.path.join(workdir, f"{base}_24k.wav")
                run(
                    f"ffmpeg -y -i {shlex.quote(path)} -ar 24000 -ac 1 {shlex.quote(converted)}"
                )
                converted_refs.append(converted)

            if len(converted_refs) == 1:
                ref_candidate = converted_refs[0]
            else:
                ref_candidate = os.path.join(workdir, "ref_concat.wav")
                concat_audio(converted_refs, ref_candidate)

            ref_path = os.path.join(workdir, "ref.wav")
            ref_duration = ffprobe_duration(ref_candidate)
            target_ref_duration = 6.0 if ref_duration < 6.0 else ref_duration
            trim_or_pad_to_duration(
                ref_candidate, ref_path, target_ref_duration, ar=24000
            )

            target_lang = (
                segment_payload.get("target_lang")
                or payload.get("target_lang")
                or self.default_target_lang
            )

            raw_tts_path = os.path.join(workdir, "tts_raw.wav")
            try:
                synthesize(
                    text,
                    ref_path,
                    language=target_lang,
                    out_path=raw_tts_path,
                    model_name=TTS_MODEL,
                )
            except Exception as exc:  # pylint: disable=broad-except
                raise JobProcessingError(f"TTS synthesis failed: {exc}") from exc

            start_point = float(segment_payload.get("start_point", 0.0) or 0.0)
            end_point = float(segment_payload.get("end_point", 0.0) or 0.0)
            slot = (
                max(0.05, end_point - start_point) if end_point > start_point else None
            )
            sub_length = segment_payload.get("sub_length")
            if sub_length is not None:
                try:
                    sub_length = float(sub_length)
                except (TypeError, ValueError):
                    sub_length = None

            target_duration = sub_length if sub_length and sub_length > 0 else slot

            final_tts_path = raw_tts_path
            if target_duration and target_duration > 0:
                fitted_path = os.path.join(workdir, "tts_fitted.wav")
                trim_or_pad_to_duration(
                    raw_tts_path, fitted_path, target_duration, ar=24000
                )
                final_tts_path = fitted_path

            # Align TTS loudness to original 화자 평균 볼륨
            ref_db = detect_mean_volume_db(ref_path)
            tts_db = detect_mean_volume_db(final_tts_path)
            gain_db_applied: Optional[float] = None
            if ref_db is not None and tts_db is not None:
                gain_db = ref_db - tts_db
                if abs(gain_db) >= 0.5:
                    gain_db = max(min(gain_db, 18.0), -12.0)
                    boosted_path = os.path.join(workdir, "tts_gain.wav")
                    apply_gain_db(final_tts_path, boosted_path, gain_db, ar=24000, ac=1)
                    os.replace(boosted_path, final_tts_path)
                    gain_db_applied = gain_db

            # 48k 변환 (mix 단계와 동일한 샘플레이트 사용)
            tts_48k_path = os.path.join(workdir, "tts_48k.wav")
            run(
                f"ffmpeg -y -i {shlex.quote(final_tts_path)} -ar 48000 -ac 1 {shlex.quote(tts_48k_path)}"
            )

            tts_key = assets.get("tts_key")
            if not tts_key and assets_prefix:
                tts_key = f"{assets_prefix.rstrip('/')}/{base_name}_tts.wav"
            if not tts_key:
                raise JobProcessingError("Unable to determine TTS upload key")

            self._upload_file(tts_48k_path, tts_key, "audio/wav")

            mix_key = assets.get("mix_key")
            bgm_key = assets.get("bgm_key")
            uploaded_mix_key = None

            if bgm_key:
                local_bgm = os.path.join(workdir, "bgm.wav")
                self.s3.download_file(self.bucket, bgm_key, local_bgm)

                aligned_bgm = local_bgm
                if target_duration and target_duration > 0:
                    bgm_duration = ffprobe_duration(local_bgm)
                    aligned_bgm = os.path.join(workdir, "bgm_aligned.wav")
                    if abs(bgm_duration - target_duration) <= 0.01:
                        shutil.copyfile(local_bgm, aligned_bgm)
                    elif bgm_duration > target_duration + 0.01:
                        run(
                            " ".join(
                                [
                                    "ffmpeg -y",
                                    f"-i {shlex.quote(local_bgm)}",
                                    f'-af "atrim=0:{target_duration:.6f},asetpts=PTS-STARTPTS"',
                                    "-ar 48000",
                                    "-ac 2",
                                    shlex.quote(aligned_bgm),
                                ]
                            )
                        )
                    else:
                        pad = max(0.0, target_duration - bgm_duration)
                        run(
                            " ".join(
                                [
                                    "ffmpeg -y",
                                    f"-i {shlex.quote(local_bgm)}",
                                    f'-filter_complex "apad=pad_dur={pad:.6f},atrim=0:{target_duration:.6f},asetpts=PTS-STARTPTS"',
                                    "-ar 48000",
                                    "-ac 2",
                                    shlex.quote(aligned_bgm),
                                ]
                            )
                        )

                bgm_gain = (
                    segment_payload.get("bgm_gain") or assets.get("bgm_gain") or 1.4
                )
                tts_mix_gain = (
                    segment_payload.get("tts_gain") or assets.get("tts_gain") or 1.0
                )
                try:
                    bgm_gain = float(bgm_gain)
                except (TypeError, ValueError):
                    bgm_gain = 1.0
                try:
                    tts_mix_gain = float(tts_mix_gain)
                except (TypeError, ValueError):
                    tts_mix_gain = 1.0

                local_mix = os.path.join(workdir, "mix.wav")
                mix_filter = (
                    f"[0:a]volume={bgm_gain:.4f}[bgm];"
                    f"[1:a]volume={tts_mix_gain:.4f}[tts];"
                    "[bgm][tts]amix=inputs=2:duration=longest:dropout_transition=0:normalize=0[m];"
                    "[m]alimiter=limit=0.95,asetpts=PTS-STARTPTS"
                )
                run(
                    " ".join(
                        [
                            "ffmpeg -y",
                            f"-i {shlex.quote(aligned_bgm)}",
                            f"-i {shlex.quote(tts_48k_path)}",
                            f'-filter_complex "{mix_filter}"',
                            "-ar 48000",
                            "-ac 2",
                            shlex.quote(local_mix),
                        ]
                    )
                )

                upload_key = mix_key or (
                    f"{assets_prefix.rstrip('/')}/{base_name}_mix.wav"
                    if assets_prefix
                    else None
                )
                if upload_key:
                    self._upload_file(local_mix, upload_key, "audio/wav")
                    uploaded_mix_key = upload_key
            else:
                raise JobProcessingError(
                    "segment_tts requires bgm_key to generate mixed audio"
                )

            metadata: Dict[str, Any] = {
                "stage": "segment_tts_completed",
                "segment": {
                    "segment_index": segment_index,
                    "segment_id": segment_id,
                    "translate_context": text,
                    "tts_key": tts_key,
                },
            }
            if uploaded_mix_key:
                metadata["segment"]["mix_key"] = uploaded_mix_key
            if gain_db_applied is not None:
                metadata["segment"]["tts_gain_db"] = gain_db_applied
            if assets_prefix:
                metadata["segment_assets_prefix"] = assets_prefix
            if project_id:
                metadata["project_id"] = project_id

            self._post_status(callback_url, "done", metadata=metadata)

        except (BotoCoreError, ClientError) as exc:
            failure = JobProcessingError(f"AWS client error: {exc}")
            self._safe_fail(callback_url, failure)
            raise failure
        except JobProcessingError as exc:
            self._safe_fail(callback_url, exc)
            raise
        except Exception as exc:  # pylint: disable=broad-except
            wrapped = JobProcessingError(str(exc))
            self._safe_fail(callback_url, wrapped)
            raise wrapped

    def _handle_segment_mix(self, payload: Dict[str, Any]) -> None:
        job_id = payload.get("job_id")
        callback_url = payload.get("callback_url")
        if not job_id or not callback_url:
            raise JobProcessingError("segment_mix requires job_id and callback_url")
        segments: List[Dict[str, Any]] = payload.get("segments") or []
        if not segments:
            raise JobProcessingError("segment_mix requires at least one segment entry")
        intermediate_prefix = payload.get("intermediate_prefix")
        output_prefix = payload.get("output_prefix") or intermediate_prefix
        project_id = payload.get("project_id")
        workdir = os.path.join(_ensure_workdir(job_id), "segment_mix")
        os.makedirs(workdir, exist_ok=True)
        try:
            self._post_status(
                callback_url,
                "in_progress",
                metadata={
                    "stage": "segment_mix_started",
                    "job_id": job_id,
                    "project_id": project_id,
                    "count": len(segments),
                },
            )

            mix_results: List[Dict[str, Any]] = []
            for segment in segments:
                index = segment.get("index")
                if index is None:
                    raise JobProcessingError("segment entry missing index")

                assets = segment.get("assets") or {}
                bgm_key = segment.get("bgm_key") or assets.get("bgm_key")
                tts_key = segment.get("tts_key") or assets.get("tts_key")
                if not bgm_key or not tts_key:
                    if not intermediate_prefix:
                        raise JobProcessingError(
                            "segment entry missing S3 keys and no intermediate_prefix provided"
                        )
                    prefix = intermediate_prefix.rstrip("/")
                    bgm_key = bgm_key or f"{prefix}/{index}_bgm.wav"
                    tts_key = tts_key or f"{prefix}/{index}_tts.wav"

                target_prefix = output_prefix or intermediate_prefix
                if not target_prefix:
                    target_prefix = os.path.dirname(bgm_key)
                target_prefix = target_prefix.rstrip("/")
                output_key = (
                    segment.get("output_key")
                    or assets.get("mix_key")
                    or f"{target_prefix}/{index}_mix.wav"
                )

                local_bgm = os.path.join(workdir, f"{index}_bgm.wav")
                local_tts = os.path.join(workdir, f"{index}_tts.wav")

                mixed_path = os.path.join(workdir, f"{index}_mix.wav")

                logger.info(
                    "Processing segment %s for job %s (bgm=%s, tts=%s -> %s)",
                    index,
                    job_id,
                    bgm_key,
                    tts_key,
                    output_key,
                )

                self.s3.download_file(self.bucket, bgm_key, local_bgm)
                self.s3.download_file(self.bucket, tts_key, local_tts)

                bgm_gain = float(segment.get("bgm_gain", 1.0))
                tts_gain = float(segment.get("tts_gain", 1.0))

                ffmpeg_cmd = (
                    "ffmpeg -y "
                    f"-i {shlex.quote(local_tts)} "
                    f"-i {shlex.quote(local_bgm)} "
                    "-filter_complex "
                    f'"[0:a]volume={tts_gain}[v0];[1:a]volume={bgm_gain}[v1];[v0][v1]amix=inputs=2:duration=longest" '
                    f"-c:a pcm_s16le {shlex.quote(mixed_path)}"
                )
                run(ffmpeg_cmd)

                self.s3.upload_file(mixed_path, self.bucket, output_key)

                mix_results.append(
                    {
                        "index": index,
                        "bgm_key": bgm_key,
                        "tts_key": tts_key,
                        "output_key": output_key,
                        "bgm_gain": bgm_gain,
                        "tts_gain": tts_gain,
                    }
                )

            self._post_status(
                callback_url,
                "done",
                metadata={
                    "stage": "segment_mix_completed",
                    "job_id": job_id,
                    "project_id": project_id,
                    "segments": mix_results,
                },
            )
        except (BotoCoreError, ClientError) as exc:
            failure = JobProcessingError(f"AWS client error: {exc}")
            self._safe_fail(callback_url, failure)
            raise failure
        except JobProcessingError as exc:
            self._safe_fail(callback_url, exc)
            raise
        except Exception as exc:  # pylint: disable=broad-except
            wrapped = JobProcessingError(str(exc))
            self._safe_fail(callback_url, wrapped)
            raise wrapped

    def _safe_fail(self, callback_url: str, error: JobProcessingError) -> None:
        try:
            self._post_status(
                callback_url,
                "failed",
                error=str(error),
                metadata={"stage": "failed"},
            )
        except JobProcessingError as callback_exc:
            logger.error("Failed to deliver failure callback: %s", callback_exc)

    def _post_status(
        self,
        callback_url: str,
        status: str,
        *,
        result_key: Optional[str] = None,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        payload: Dict[str, Any] = {"status": status}
        if result_key is not None:
            payload["result_key"] = result_key
        if error is not None:
            payload["error"] = error
        if metadata is not None:
            payload["metadata"] = metadata

        target_url = self._normalize_callback_url(callback_url)

        try:
            resp = self.http.post(target_url, json=payload, timeout=30)
        except requests.RequestException as exc:
            raise JobProcessingError(f"Callback request failed: {exc}") from exc

        if not resp.ok:
            raise JobProcessingError(
                f"Callback responded with {resp.status_code}: {resp.text[:200]}"
            )

    def _normalize_callback_url(self, callback_url: str) -> str:
        """
        Replace localhost callback hosts with a reachable hostname when running inside containers.
        """
        parsed = urlparse(callback_url)
        if (
            parsed.hostname in {"localhost", "127.0.0.1"}
            and self.callback_localhost_host
        ):
            host = self.callback_localhost_host
            netloc = host
            if parsed.port:
                netloc = f"{host}:{parsed.port}"
            parsed = parsed._replace(netloc=netloc)
        return urlunparse(parsed)


def run_worker() -> None:
    worker = QueueWorker()
    worker.poll_forever()


if __name__ == "__main__":
    run_worker()
