# app/tts.py
import os
from functools import lru_cache

LANG_MAP = {"en": "en", "ja": "ja", "ko": "ko"}
TTS_DEVICE = os.getenv("TTS_DEVICE", "cpu").lower()

def _prepare_torch_deserialization():
    """
    PyTorch 안전 직렬화 허용목록에 XTTS 관련 클래스를 추가.
    (필요 객체가 더 나오면 여기서 계속 추가 가능)
    """
    try:
        import torch.serialization as ts
        allow = []
        try:
            from TTS.tts.configs.xtts_config import XttsConfig
            allow.append(XttsConfig)
        except Exception:
            pass
        try:
            from TTS.tts.models.xtts import XttsAudioConfig
            allow.append(XttsAudioConfig)
        except Exception:
            pass
        if allow:
            ts.add_safe_globals(allow)
    except Exception:
        pass

@lru_cache(maxsize=1)
def _get_tts(model_name: str):
    _prepare_torch_deserialization()

    # 1차 시도: 안전 직렬화(allowlist 기반)만으로 로드
    from TTS.api import TTS
    import torch
    try:
        tts = TTS(model_name)
    except Exception as e:
        # 2차 폴백: 신뢰하는 체크포인트에 한해 weights_only=False로 재시도
        if "Weights only load failed" in str(e):
            orig_load = torch.load
            def _patched_load(*args, **kwargs):
                kwargs.setdefault("weights_only", False)  # 안전장치 완화 (공식 XTTS만 쓸 때)
                return orig_load(*args, **kwargs)
            try:
                torch.load = _patched_load
                tts = TTS(model_name)
            finally:
                torch.load = orig_load
        else:
            raise

    # 디바이스 이동
    use_cuda = (TTS_DEVICE == "cuda") and torch.cuda.is_available()
    try:
        tts.to("cuda" if use_cuda else "cpu")
    except Exception:
        pass
    return tts

def synthesize(text: str, ref_wav: str, language: str, out_path: str, model_name: str):
    tts = _get_tts(model_name)
    lang = LANG_MAP.get(language, "en")
    tts.tts_to_file(text=text, speaker_wav=ref_wav, language=lang, file_path=out_path)
