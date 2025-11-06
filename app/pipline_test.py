from .models.whisper_loader import get_model as get_whisper

def _whisper_transcribe(audio_wav_16k: str):
    model = get_whisper()
    segments, _ = model.transcribe(
        audio_wav_16k,
        language=None,
        vad_filter=True,
        word_timestamps=False,
    )
    return [
        {"start": float(s.start), "end": float(s.end), "text": s.text.strip()}
        for s in segments
    ]