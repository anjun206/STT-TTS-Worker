from functools import lru_cache
from pathlib import Path
import os
import tarfile
import boto3
from botocore.exceptions import ClientError
from faster_whisper import WhisperModel

LOCAL_BASE = Path(os.getenv("WHISPER_LOCAL_DIR", "/app/models/whisper"))
S3_BUCKET = os.getenv("WHISPER_S3_BUCKET")
S3_PREFIX = os.getenv("WHISPER_S3_PREFIX", "whisper")
MODEL_NAME = os.getenv("WHISPER_MODEL", "small")

def _extract_tar(tar_path: Path, dst_dir: Path):
    with tarfile.open(tar_path, "r:gz") as tf:
        tf.extractall(dst_dir)

def _ensure_local_model(model_name: str) -> Path:
    target_dir = LOCAL_BASE / model_name
    marker = target_dir / ".ready"
    if marker.exists():
        return target_dir

    tmp_dir = target_dir.with_suffix(".downloading")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    client = boto3.client("s3")
    key = f"{S3_PREFIX}/{model_name}.tar.gz"
    tmp_tar = tmp_dir / "model.tar.gz"

    try:
        client.download_file(S3_BUCKET, key, str(tmp_tar))
        _extract_tar(tmp_tar, tmp_dir)
        tmp_tar.unlink()
        tmp_dir.rename(target_dir)
        marker.touch()
    except ClientError as exc:
        raise RuntimeError(f"Failed to fetch model {model_name} from S3: {exc}") from exc

    return target_dir

@lru_cache(maxsize=1)
def get_model():
    device = "cuda" if os.getenv("USE_GPU", "1") == "1" else "cpu"
    compute = "float16" if device == "cuda" else "int8"
    local_path = _ensure_local_model(MODEL_NAME)
    return WhisperModel(local_path, device=device, compute_type=compute)