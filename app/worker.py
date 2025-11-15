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
from configs import JobPaths, ensure_job_dirs
from services.demucs_split import split_vocals
from services.tts import _transcribe_prompt_text, _synthesize_with_cosyvoice2
from pydub import AudioSegment
import shutil

# 로깅 설정
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
noisy_loggers = ["boto3", "botocore", "s3transfer", "urllib3"]
for logger_name in noisy_loggers:
    logging.getLogger(logger_name).setLevel(logging.WARNING)

# AWS 설정
AWS_REGION = os.getenv("AWS_REGION", "ap-northeast-2")
JOB_QUEUE_URL = os.getenv("JOB_QUEUE_URL")
AWS_S3_BUCKET = os.getenv("AWS_S3_BUCKET")

if not all([JOB_QUEUE_URL, AWS_S3_BUCKET]):
    raise ValueError(
        "JOB_QUEUE_URL and AWS_S3_BUCKET environment variables must be set."
    )

sqs_client = boto3.client("sqs", region_name=AWS_REGION)
s3_client = boto3.client("s3", region_name=AWS_REGION)


def send_callback(
    callback_url: str,
    status: str,
    message: str,
    stage: str | None = None,
    metadata: dict | None = None,
):
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
        logging.info(
            f"Sent callback to {callback_url} with status: {status}, stage: {stage or 'N/A'}"
        )
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


def resolve_output_prefix(
    project_id: str | None, job_id: str, override: str | None
) -> str:
    """결과물을 저장할 기본 경로를 계산합니다."""
    if override:
        return override.rstrip("/")
    if project_id:
        return f"projects/{project_id}/outputs/{job_id}"
    return f"jobs/{job_id}/outputs"


def upload_metadata_to_s3(bucket: str, key: str, metadata: dict) -> bool:
    """파이프라인 메타데이터를 JSON으로 직렬화해 S3에 업로드합니다."""
    try:
        body = json.dumps(metadata, ensure_ascii=False).encode("utf-8")
        logging.info(f"Uploading metadata to s3://{bucket}/{key}...")
        s3_client.put_object(
            Bucket=bucket,
            Key=key,
            Body=body,
            ContentType="application/json",
        )
        logging.info(f"Successfully uploaded metadata to s3://{bucket}/{key}")
        return True
    except ClientError as e:
        logging.error(f"Failed to upload metadata to S3: {e}")
        return False


def _resolve_s3_location(raw: str, default_bucket: str) -> tuple[str, str]:
    """
    Parse strings like 's3://bucket/key' or bare keys into (bucket, key).
    Falls back to default_bucket when explicit bucket is missing.
    """
    value = (raw or "").strip()
    if not value:
        raise ValueError("빈 S3 위치 문자열입니다.")
    if value.startswith("s3://"):
        remainder = value[5:]
        if "/" not in remainder:
            raise ValueError(f"Invalid S3 URI: {raw}")
        bucket, key = remainder.split("/", 1)
        return bucket, key
    key = value.lstrip("/")
    return default_bucket, key


def _sync_segment_to_range(
    input_path: Path, target_duration_ms: int, output_path: Path
) -> Path:
    """
    Coerce a single TTS clip to the requested duration by trimming or padding silence.
    """
    if target_duration_ms <= 0:
        raise ValueError("target_duration_ms must be positive")

    audio = AudioSegment.from_file(str(input_path))
    current_ms = len(audio)
    if current_ms == target_duration_ms:
        if input_path != output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(input_path, output_path)
        return output_path

    if current_ms > target_duration_ms:
        processed = audio[:target_duration_ms]
    else:
        silence = AudioSegment.silent(duration=target_duration_ms - current_ms)
        processed = audio + silence

    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed.export(str(output_path), format="wav")
    return output_path


def _segments_with_remote_audio_paths(
    segments: list[dict],
    project_prefix: str,
    job_id: str,
    paths: JobPaths,
) -> list[dict]:
    """
    Copy segment dicts while rewriting local `/data/interim/<job_id>` audio paths
    to remote keys that mirror the uploaded layout.
    """
    if not segments:
        return []
    base_dir = paths.interim_dir
    remote_prefix = f"{project_prefix}/interim/{job_id}"
    normalized: list[dict] = []
    for segment in segments:
        updated = dict(segment)
        audio_value = updated.get("audio_file")
        if isinstance(audio_value, str):
            if audio_value.startswith("s3://") or audio_value.startswith(remote_prefix):
                normalized.append(updated)
                continue
            candidate = Path(audio_value)
            try:
                relative_path = candidate.relative_to(base_dir)
            except ValueError:
                logging.debug(
                    "audio_file 경로 %s 가 %s 기준 상대 경로가 아닙니다. 원본 값을 유지합니다.",
                    candidate,
                    base_dir,
                )
            else:
                updated["audio_file"] = f"{remote_prefix}/{relative_path.as_posix()}"
        normalized.append(updated)
    return normalized


def _build_speaker_metadata(
    paths: JobPaths, project_prefix: str, job_id: str
) -> list[dict]:
    """
    Collect speaker metadata consisting of speaker name, uploaded sample key,
    and optional prompt text.
    """
    speaker_refs_path = paths.vid_tts_dir / "speaker_refs.json"
    if not speaker_refs_path.is_file():
        return []
    try:
        refs = json.loads(speaker_refs_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logging.warning("Failed to parse %s: %s", speaker_refs_path, exc)
        return []

    remote_prefix = f"{project_prefix}/interim/{job_id}"
    base_dir = paths.interim_dir.resolve()
    metadata: list[dict] = []

    for speaker, payload in refs.items():
        if isinstance(payload, str):
            audio_value = payload
            prompt_text = ""
        elif isinstance(payload, dict):
            audio_value = payload.get("audio") or payload.get("path") or ""
            prompt_text = (payload.get("text") or "").strip()
        else:
            continue

        if not audio_value:
            continue

        sample_path = Path(audio_value)
        if not sample_path.is_absolute():
            sample_path = (paths.vid_tts_dir / sample_path).resolve()

        try:
            rel_path = sample_path.relative_to(base_dir)
            voice_sample_key = f"{remote_prefix}/{rel_path.as_posix()}"
        except ValueError:
            logging.warning(
                "Voice sample %s is outside interim dir; using absolute path.",
                sample_path,
            )
            voice_sample_key = str(sample_path)

        entry = {
            "speaker": speaker,
            "voice_sample_key": voice_sample_key,
        }
        if prompt_text:
            entry["prompt_text"] = prompt_text
        metadata.append(entry)

    return metadata


def _parse_positive_int(value, field_name: str) -> int | None:
    """Optional int parser that tolerates strings and invalid inputs."""
    if value is None or value == "":
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        logging.warning(
            "Ignoring %s=%r because it is not an integer", field_name, value
        )
        return None
    if parsed < 1:
        logging.warning("Ignoring %s=%r because it must be >= 1", field_name, value)
        return None
    return parsed


def full_pipeline(job_details: dict):
    """전체 더빙 파이프라인을 실행합니다."""
    job_id = job_details["job_id"]
    project_id = job_details.get("project_id")
    input_key = job_details["input_key"]
    callback_url = job_details["callback_url"]

    target_lang = job_details.get("target_lang", "en")
    source_lang = job_details.get("source_lang")
    speaker_count = _parse_positive_int(
        job_details.get("speaker_count"), "speaker_count"
    )
    voice_config = job_details.get("voice_config")
    input_bucket = job_details.get("input_bucket") or AWS_S3_BUCKET
    output_bucket = job_details.get("output_bucket") or AWS_S3_BUCKET
    project_prefix = f"projects/{project_id}" if project_id else "jobs"
    output_prefix = resolve_output_prefix(
        project_id, job_id, job_details.get("output_prefix")
    )
    metadata_key = (
        job_details.get("metadata_key") or f"{output_prefix}/metadata/{job_id}.json"
    )

    send_callback(
        callback_url,
        "in_progress",
        f"Starting full pipeline for job {job_id}",
        stage="starting",
        metadata={
            "job_id": job_id,
            "project_id": project_id,
            "target_lang": target_lang,
        },
    )

    # 1. 로컬 작업 디렉토리 설정
    paths = ensure_job_dirs(job_id)
    source_video_path = paths.input_dir / Path(input_key).name

    # 2. S3에서 원본 영상 다운로드
    if not download_from_s3(input_bucket, input_key, source_video_path):
        send_callback(
            callback_url,
            "failed",
            "Failed to download source video from S3.",
            stage="download_failed",
            metadata={"job_id": job_id, "project_id": project_id},
        )
        return

    # 3. voice_config에서 사용자 음성 샘플 다운로드 (필요 시)
    user_voice_sample_path = None
    if voice_config and voice_config.get("kind") == "s3" and voice_config.get("key"):
        voice_key = voice_config["key"]
        voice_bucket = (
            voice_config.get("bucket")
            or voice_config.get("bucket_name")
            or input_bucket
        )
        user_voice_sample_path = paths.interim_dir / Path(voice_key).name
        if not download_from_s3(voice_bucket, voice_key, user_voice_sample_path):
            send_callback(
                callback_url,
                "failed",
                f"Failed to download voice sample from S3 key: {voice_key}",
                stage="download_failed",
                metadata={"job_id": job_id, "project_id": project_id},
            )
            return
        send_callback(
            callback_url,
            "in_progress",
            "Custom voice sample downloaded.",
            stage="downloaded",
            metadata={"job_id": job_id, "project_id": project_id},
        )

    translations: list[dict] = []
    segments_payload: list[dict] = []
    speaker_metadata: list[dict] = []
    final_audio_path: Path | None = None

    try:
        # 4. ASR (STT)
        send_callback(
            callback_url, "in_progress", "Starting ASR...", stage="asr_started"
        )
        run_asr(
            job_id,
            source_video_path,
            source_lang=source_lang,
            speaker_count=speaker_count,
        )
        # ASR 결과물(compact transcript)을 S3에 업로드
        from services.transcript_store import COMPACT_ARCHIVE_NAME

        asr_result_path = paths.src_sentence_dir / COMPACT_ARCHIVE_NAME
        upload_to_s3(
            output_bucket,
            f"{project_prefix}/interim/{job_id}/{COMPACT_ARCHIVE_NAME}",
            asr_result_path,
        )
        # 원본 오디오(audio.wav)를 S3에 업로드
        raw_audio_path = paths.vid_speaks_dir / "audio.wav"
        audio_key = None
        if raw_audio_path.is_file():
            audio_key = f"{project_prefix}/interim/{job_id}/audio/audio.wav"
            if upload_to_s3(output_bucket, audio_key, raw_audio_path):
                logging.info(f"Raw audio uploaded to s3://{output_bucket}/{audio_key}")
            else:
                logging.warning("Failed to upload audio.wav to S3")

        # 발화 음성(vocals.wav)과 배경음(background.wav)을 S3에 업로드
        vocals_path = paths.vid_speaks_dir / "vocals.wav"
        background_path = paths.vid_bgm_dir / "background.wav"

        vocals_key = None
        background_key = None

        if vocals_path.is_file():
            vocals_key = f"{project_prefix}/interim/{job_id}/audio/vocals.wav"
            if upload_to_s3(output_bucket, vocals_key, vocals_path):
                logging.info(f"Vocals uploaded to s3://{output_bucket}/{vocals_key}")
            else:
                logging.warning("Failed to upload vocals.wav to S3")

        if background_path.is_file():
            background_key = f"{project_prefix}/interim/{job_id}/audio/background.wav"
            if upload_to_s3(output_bucket, background_key, background_path):
                logging.info(
                    f"Background uploaded to s3://{output_bucket}/{background_key}"
                )
            else:
                logging.warning("Failed to upload background.wav to S3")

        send_callback(
            callback_url,
            "in_progress",
            "ASR completed.",
            stage="asr_completed",
            metadata=(
                {
                    "audio_key": audio_key,
                    "vocals_key": vocals_key,
                    "background_key": background_key,
                }
                if (audio_key or vocals_key or background_key)
                else None
            ),
        )

        # 5. 번역
        send_callback(
            callback_url,
            "in_progress",
            "Starting translation...",
            stage="translation_started",
        )
        translations = translate_transcript(job_id, target_lang, src_lang=source_lang)
        # 번역 결과물(translated.json)을 S3에 업로드
        trans_result_path = paths.trg_sentence_dir / "translated.json"
        upload_to_s3(
            output_bucket,
            f"{project_prefix}/interim/{job_id}/translated.json",
            trans_result_path,
        )
        send_callback(
            callback_url,
            "in_progress",
            "Translation completed.",
            stage="translation_completed",
        )

        # 6. TTS
        send_callback(
            callback_url, "in_progress", "Starting TTS...", stage="tts_started"
        )
        segments_payload = generate_tts(
            job_id, target_lang, voice_sample_path=user_voice_sample_path
        )
        # TTS 결과물(개별 wav 파일 및 segments.json)을 S3에 업로드
        tts_dir = paths.vid_tts_dir
        for tts_file in tts_dir.glob("**/*"):
            if not tts_file.is_file():
                continue
            try:
                relative_path = tts_file.relative_to(paths.interim_dir)
            except ValueError:
                relative_path = tts_file.relative_to(tts_dir)
                logging.warning(
                    "TTS 경로 %s 를 interim 디렉터리 기준으로 계산하지 못했습니다. "
                    "tts 디렉터리 상대 경로를 사용합니다.",
                    tts_file,
                )
            tts_key = f"{project_prefix}/interim/{job_id}/{relative_path}"
            upload_to_s3(output_bucket, str(tts_key), tts_file)
        speaker_metadata = _build_speaker_metadata(paths, project_prefix, job_id)
        send_callback(
            callback_url,
            "in_progress",
            "TTS completed.",
            stage="tts_completed",
            metadata={
                "job_id": job_id,
                "project_id": project_id,
                "speakers": speaker_metadata,
                "speaker_count": len(speaker_metadata),
            },
        )

        # 7. Sync
        send_callback(
            callback_url, "in_progress", "Starting sync...", stage="sync_started"
        )
        try:
            synced_segments = sync_segments(job_id)
        except FileNotFoundError as exc:
            logging.info(
                "Sync artifacts not found, skipping segment alignment: %s", exc
            )
            synced_segments = []
        except Exception as exc:  # pylint: disable=broad-except
            logging.warning("Sync step failed, proceeding without alignment: %s", exc)
            synced_segments = []
        else:
            if synced_segments:
                segments_payload = synced_segments
        # Sync 결과물(synced 디렉토리)을 S3에 업로드
        synced_dir = paths.vid_tts_dir / "synced"
        for sync_file in synced_dir.glob("**/*"):
            if not sync_file.is_file():
                continue
            try:
                relative_path = sync_file.relative_to(paths.interim_dir)
            except ValueError:
                relative_path = sync_file.relative_to(synced_dir)
                logging.warning(
                    "Synced 경로 %s 를 interim 디렉터리 기준으로 계산하지 못했습니다. "
                    "synced 디렉터리 상대 경로를 사용합니다.",
                    sync_file,
                )
            sync_key = f"{project_prefix}/interim/{job_id}/{relative_path}"
            upload_to_s3(output_bucket, str(sync_key), sync_file)
        send_callback(
            callback_url, "in_progress", "Sync completed.", stage="sync_completed"
        )

        # 8. Mux
        send_callback(
            callback_url, "in_progress", "Starting mux...", stage="mux_started"
        )
        mux_results = mux_audio_video(job_id, source_video_path)
        output_video_path = Path(mux_results["output_video"])
        final_audio_path = Path(mux_results["output_audio"])

        # 9. 최종 결과물 S3에 업로드
        result_key = (
            job_details.get("result_key") or f"{output_prefix}/{output_video_path.name}"
        )
        if not upload_to_s3(output_bucket, result_key, output_video_path):
            raise Exception("Failed to upload final video to S3")

        metadata_segments = _segments_with_remote_audio_paths(
            segments_payload,
            project_prefix,
            job_id,
            paths,
        )

        metadata_payload = {
            "job_id": job_id,
            "project_id": project_id,
            "target_lang": target_lang,
            "source_lang": source_lang,
            "input_bucket": input_bucket,
            "input_key": input_key,
            "result_bucket": output_bucket,
            "result_key": result_key,
            "metadata_key": metadata_key,
            "segments": metadata_segments,
            "segment_count": len(metadata_segments),
            "translations": translations,
            "speakers": speaker_metadata,
            "speaker_count": len(speaker_metadata),
        }
        if final_audio_path:
            metadata_payload["audio_artifact"] = str(final_audio_path)

        if not upload_metadata_to_s3(output_bucket, metadata_key, metadata_payload):
            raise Exception("Failed to upload metadata to S3")

        final_metadata = {
            "job_id": job_id,
            "project_id": project_id,
            "result_bucket": output_bucket,
            "result_key": result_key,
            "metadata_key": metadata_key,
            "segment_count": len(metadata_segments),
            "speaker_count": len(speaker_metadata),
            "target_lang": target_lang,
        }
        if source_lang:
            final_metadata["source_lang"] = source_lang

        send_callback(
            callback_url,
            "done",
            "Pipeline completed successfully.",
            stage="done",
            metadata=final_metadata,
        )

    except Exception as e:
        logging.error(f"Pipeline failed for job {job_id}: {e}", exc_info=True)
        send_callback(
            callback_url,
            "failed",
            str(e),
            stage="failed",
            metadata={"job_id": job_id, "project_id": project_id},
        )


def _handle_tts_segments(job_details: dict) -> None:
    """segment_tts / tts 작업을 처리합니다."""
    job_id = job_details.get("job_id")
    callback_url = job_details.get("callback_url")
    segments_req = job_details.get("segments") or []

    if not job_id or not callback_url:
        raise ValueError("segment_tts requires both job_id and callback_url.")
    if not segments_req:
        raise ValueError("segment_tts requires at least one segment entry.")

    project_id = job_details.get("project_id")
    target_lang = job_details.get("target_lang", "ko")
    mod_raw = (job_details.get("mod") or "dynamic").strip().lower()
    mod = mod_raw if mod_raw in {"fixed", "dynamic"} else "dynamic"
    output_bucket = job_details.get("output_bucket") or AWS_S3_BUCKET
    project_prefix = f"projects/{project_id}" if project_id else "jobs"
    remote_interim_prefix = f"{project_prefix}/interim/{job_id}"

    send_callback(
        callback_url,
        "in_progress",
        f"Starting segment TTS for job {job_id}",
        stage="segment_tts_started",
        metadata={"job_id": job_id, "project_id": project_id, "mod": mod},
    )

    try:
        paths = ensure_job_dirs(job_id)
        resynth_dir = paths.vid_tts_dir / "resynth"
        resynth_dir.mkdir(parents=True, exist_ok=True)

        speaker_spec = job_details.get("speaker_voices") or {}
        voice_key = speaker_spec.get("key")
        if not voice_key:
            raise ValueError("speaker_voices.key is required for segment_tts.")
        speaker_bucket = (
            speaker_spec.get("bucket")
            or job_details.get("input_bucket")
            or output_bucket
        )

        speaker_asset_dir = paths.interim_dir / "speaker_assets"
        speaker_asset_dir.mkdir(parents=True, exist_ok=True)

        voice_bucket, resolved_voice_key = _resolve_s3_location(
            voice_key, speaker_bucket
        )
        sample_path = speaker_asset_dir / Path(resolved_voice_key).name
        if not download_from_s3(voice_bucket, resolved_voice_key, sample_path):
            raise RuntimeError(
                f"Failed to download speaker sample from {resolved_voice_key}"
            )

        prompt_text = (speaker_spec.get("text_prompt_value") or "").strip()
        text_prompt_key = speaker_spec.get("text_prompt")
        if not prompt_text and text_prompt_key:
            prompt_bucket = (
                speaker_spec.get("text_prompt_bucket")
                or speaker_spec.get("bucket")
                or job_details.get("input_bucket")
                or output_bucket
            )
            prompt_bucket, prompt_key = _resolve_s3_location(
                text_prompt_key, prompt_bucket
            )
            prompt_path = speaker_asset_dir / Path(prompt_key).name
            if not download_from_s3(prompt_bucket, prompt_key, prompt_path):
                raise RuntimeError(
                    f"Failed to download speaker text prompt from {prompt_key}"
                )
            prompt_text = prompt_path.read_text(encoding="utf-8").strip()

        if not prompt_text:
            fallback_prompt = (job_details.get("prompt_text") or "").strip()
            prompt_text = fallback_prompt or _transcribe_prompt_text(sample_path)

        if not prompt_text:
            raise ValueError("Unable to resolve prompt text for TTS segments.")

        results: list[dict] = []
        for idx, seg_req in enumerate(segments_req):
            text = (seg_req.get("text") or "").strip()
            if not text:
                raise ValueError(f"Segment {idx} is missing 'text'.")

            s_val = seg_req.get("s", seg_req.get("start"))
            e_val = seg_req.get("e", seg_req.get("end"))

            def _to_seconds(value) -> float | None:
                if value is None or value == "":
                    return None
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return None

            s_sec = _to_seconds(s_val) or 0.0
            e_sec = _to_seconds(e_val)

            s_ms = max(0, int(s_sec * 1000))
            e_ms = int(e_sec * 1000) if e_sec is not None else None

            local_tts = resynth_dir / f"seg_{idx:04d}.wav"

            _synthesize_with_cosyvoice2(
                text=text,
                prompt_text=prompt_text,
                sample_path=sample_path,
                output_path=local_tts,
            )

            synced_path = local_tts
            if mod == "fixed":
                if e_ms is None or e_ms <= s_ms:
                    raise ValueError(f"Segment {idx} missing valid s/e for fixed mode.")
                target_duration_ms = max(1, e_ms - s_ms)
                synced_path = _sync_segment_to_range(
                    local_tts,
                    target_duration_ms,
                    resynth_dir / f"seg_{idx:04d}_synced.wav",
                )

            try:
                relative = synced_path.relative_to(paths.interim_dir)
            except ValueError:
                raise RuntimeError(
                    f"TTS artifact {synced_path} is outside interim dir"
                ) from None
            s3_key = f"{remote_interim_prefix}/{relative.as_posix()}"
            if not upload_to_s3(output_bucket, s3_key, synced_path):
                raise RuntimeError(f"Failed to upload segment {idx} to S3.")

            results.append(
                {
                    "index": idx,
                    "text": text,
                    "s": s_sec,
                    "e": e_sec,
                    "audio_key": s3_key,
                    "bucket": output_bucket,
                    "mod": mod,
                }
            )

        send_callback(
            callback_url,
            "done",
            "Segment TTS completed.",
            stage="segment_tts_completed",
            metadata={
                "job_id": job_id,
                "project_id": project_id,
                "target_lang": target_lang,
                "mod": mod,
                "segments": results,
            },
        )
    except Exception as exc:
        logging.error("segment_tts failed for job %s: %s", job_id, exc, exc_info=True)
        send_callback(
            callback_url,
            "failed",
            f"Segment TTS failed: {exc}",
            stage="segment_tts_failed",
            metadata={"job_id": job_id, "project_id": project_id, "mod": mod},
        )
        raise


def _handle_test_synthesis(job_details: dict):
    """test_synthesis 작업을 처리합니다.
    처리 순서:
    1. s3에서 보이스 샘플 다운로드
    2. Demucs로 보컬 분리 (전처리)
    3. STT로 프롬프느 텍스트 추출
    4. CosyVoice2로 TTS 생성
    5. S3에 결과 업로드
    6. 콜백 전송
    """
    job_id = job_details.get("job_id")
    callback_url = job_details.get("callback_url")
    file_path = job_details.get("file_path") or job_details.get("input_key")
    text = job_details.get("text")
    target_lang = job_details.get("target_lang", "ko")
    voice_sample_id = job_details.get("voice_sample_id")

    if not all([job_id, callback_url, file_path, text]):
        raise ValueError(
            "Missing required fields: job_id, callback_url, file_path, text"
        )

    logging.info(f"Processing test_synthesis job {job_id}")
    send_callback(
        callback_url,
        "in_progress",
        f"Starting test_synthesis for job {job_id}",
        stage="test_synthesis_started",
    )

    # 1. 작업 디렉토리 생성
    paths = ensure_job_dirs(job_id)

    # 2. S3에서 보이스 샘플 다운로드
    local_voice_sample = paths.input_dir / "voice_sample.wav"
    logging.info(
        f"Downloading voice sample from s3://{AWS_S3_BUCKET}/{file_path} to {local_voice_sample}"
    )
    if not download_from_s3(AWS_S3_BUCKET, file_path, local_voice_sample):
        send_callback(
            callback_url,
            "failed",
            f"Failed to download voice sample from S3: {file_path}",
            stage="download_failed",
        )
        return

    send_callback(
        callback_url,
        "in_progress",
        "Voice sample downloaded, starting preprocessing...",
        stage="downloaded",
    )
    send_callback(
        callback_url,
        "in_progress",
        "Voice sample downloaded, starting preprocessing...",
        stage="downloaded",
    )

    # 2-1. 오디오 파일이 30초를 넘으면 30초로 자르기
    try:
        audio = AudioSegment.from_file(str(local_voice_sample))
        max_duration_ms = 30 * 1000  # 30초 = 30000ms

        if len(audio) > max_duration_ms:
            logging.info(f"Audio file is {len(audio)/1000:.2f}s, trimming to 30s")
            audio = audio[:max_duration_ms]
            audio.export(str(local_voice_sample), format="wav")
            logging.info("Audio file trimmed to 30 seconds")
        else:
            logging.info(f"Audio file is {len(audio)/1000:.2f}s, no trimming needed")
    except Exception as e:
        logging.warning(
            f"Failed to check/trim audio duration: {e}, continuing with original file"
        )

    try:
        # 3. Demucs 전처리: 보이스 샘플을 paths.vid_speaks_dir/audio.wav에 복사
        # split_vocals는 paths.vid_speaks_dir/audio.wav를 찾으므로 복사 필요
        audio_path_for_demucs = paths.vid_speaks_dir / "audio.wav"
        audio_path_for_demucs.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(local_voice_sample, audio_path_for_demucs)

        logging.info("Running Demucs for vocal separation...")
        demucs_result = split_vocals(job_id)
        vocals_path = Path(demucs_result["vocals"])

        send_callback(
            callback_url,
            "in_progress",
            "Vocal separation completed, extracting prompt text...",
            stage="preprocessing_completed",
        )

        # 4. STT로 프롬프트 텍스트 추출
        logging.info("Extracting prompt text using STT...")
        prompt_text = _transcribe_prompt_text(vocals_path)

        if not prompt_text:
            logging.warning("Failed to extract prompt text, using empty string")
            prompt_text = ""

        logging.info(f"Extracted prompt text: {prompt_text[:100]}...")

        send_callback(
            callback_url,
            "in_progress",
            "Prompt text extracted, generating TTS...",
            stage="prompt_extracted",
        )

        # 5. TTS 생성
        output_path = paths.outputs_dir / "tts_output.wav"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logging.info(f"Generating TTS with CosyVoice2...")
        _synthesize_with_cosyvoice2(
            text=text,
            prompt_text=prompt_text,
            sample_path=vocals_path,
            output_path=output_path,
        )

        send_callback(
            callback_url,
            "in_progress",
            "TTS generated, uploading to S3...",
            stage="tts_completed",
        )

        # 6. S3에 업로드
        s3_key = f"voice-samples/tts/{job_id}.wav"
        if not upload_to_s3(AWS_S3_BUCKET, s3_key, output_path):
            raise Exception("Failed to upload TTS result to S3")

        # 7. 콜백: 완료
        audio_sample_url = f"/api/storage/media/{s3_key}"
        send_callback(
            callback_url,
            "done",
            "Test synthesis completed successfully.",
            stage="test_synthesis_completed",
            metadata={
                "result_key": s3_key,
                "audio_sample_url": audio_sample_url,
                "voice_sample_id": voice_sample_id,
                "prompt_text": prompt_text,
            },
        )
        logging.info(f"Job {job_id} completed successfully")

    except Exception as e:
        logging.error(f"Test synthesis failed for job {job_id}: {e}", exc_info=True)
        send_callback(
            callback_url,
            "failed",
            f"Test synthesis failed: {str(e)}",
            stage="test_synthesis_failed",
        )
        raise


async def poll_sqs():
    """SQS 큐를 폴링하여 메시지를 처리합니다."""
    logging.info("Starting SQS poller...")
    while True:
        try:
            response = sqs_client.receive_message(
                QueueUrl=JOB_QUEUE_URL,
                MaxNumberOfMessages=1,
                WaitTimeSeconds=20,
                MessageAttributeNames=["All"],
            )

            messages = response.get("Messages", [])
            if not messages:
                continue

            for message in messages:
                receipt_handle = message["ReceiptHandle"]
                try:
                    logging.info(f"Received message: {message['MessageId']}")
                    job_details = json.loads(message["Body"])

                    task = (job_details.get("task") or "full_pipeline").lower()
                    job_id = job_details.get("job_id", "unknown")

                    logging.info(f"Received task: {task} for job {job_id}")

                    if task == "test_synthesis":
                        _handle_test_synthesis(job_details)
                    elif task in ("segment_tts", "tts"):
                        _handle_tts_segments(job_details)
                    else:
                        # 파이프라인 실행
                        full_pipeline(job_details)

                    # 파이프라인 실행
                    # full_pipeline(job_details)

                    # 처리 완료 후 메시지 삭제
                    sqs_client.delete_message(
                        QueueUrl=JOB_QUEUE_URL, ReceiptHandle=receipt_handle
                    )
                    logging.info(f"Deleted message: {message['MessageId']}")

                except json.JSONDecodeError as e:
                    logging.error(f"Invalid JSON in message body: {e}")
                    # 잘못된 형식의 메시지는 큐에서 삭제
                    sqs_client.delete_message(
                        QueueUrl=JOB_QUEUE_URL, ReceiptHandle=receipt_handle
                    )
                except Exception as e:
                    logging.error(
                        f"Error processing message {message.get('MessageId', 'N/A')}: {e}"
                    )
                    # 처리 중 에러 발생 시, 메시지를 바로 삭제하지 않고 SQS의 Visibility Timeout에 따라 재처리되도록 둡니다.
                    # 또는 Dead Letter Queue로 보내는 정책을 사용할 수 있습니다.
                    # 여기서는 일단 로깅만 하고 넘어갑니다.
                    pass

        except (BotoCoreError, ClientError) as e:
            logging.error(f"Error polling SQS: {e}")
            await asyncio.sleep(10)  # 에러 발생 시 잠시 대기 후 재시도
        except Exception as e:
            logging.error(f"An unexpected error occurred in the poller: {e}")
            await asyncio.sleep(10)


if __name__ == "__main__":
    asyncio.run(poll_sqs())
