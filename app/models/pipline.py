# backend/app/pipeline.py 중 일부
from .models.whisper_loader import get_model as get_whisper

def _whisper_transcribe(audio_wav_16k: str):
    model = get_whisper()
    segments, info = model.transcribe(audio_wav_16k, language=None, vad_filter=True, word_timestamps=False)
