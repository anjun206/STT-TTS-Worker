# mux.py
import os
import subprocess
from pydub import AudioSegment
from config import get_job_paths
from pathlib import Path


def mux_audio_video(job_id: str, video_input_path: Path):
    """합성 음성과 배경음을 결합하고 원본 영상에 다시 입혀 최종 영상을 생성합니다."""
    paths = get_job_paths(job_id)
    background_path = paths.vid_bgm_dir / "background.wav"
    base_tts_dir = paths.vid_tts_dir
    synced_dir = base_tts_dir / "synced"
    tts_dir = synced_dir if synced_dir.is_dir() else base_tts_dir
    
    video_input = video_input_path
    if not video_input.is_file():
        raise RuntimeError(f"Original video file not found for muxing at {video_input}")
    if not background_path.is_file():
        raise FileNotFoundError("Background audio not found. Run Demucs stage.")
    if not tts_dir.is_dir():
        raise FileNotFoundError("TTS audio segments not found. Run TTS stage.")

    # 배경 오디오 로드
    background_audio = AudioSegment.from_wav(str(background_path))
    total_duration_ms = len(background_audio)
    # 배경음을 복제해 믹싱의 베이스로 사용
    final_audio = background_audio[:]  # 복제본

    # 합성된 음성 구간을 적절한 시작 위치에 오버레이
    wav_files = sorted(
        f for f in os.listdir(tts_dir) if f.lower().endswith(".wav")
    )
    if not wav_files:
        raise FileNotFoundError("TTS 오디오 파일이 존재하지 않습니다.")
    for fname in wav_files:
        if fname.endswith(".wav"):
            segment_audio = AudioSegment.from_wav(str(tts_dir / fname))
            # 파일명 끝부분에 기록된 시작 시간을 파싱
            try:
                start_time = float(os.path.splitext(fname)[0].split("_")[-1])
            except:
                start_time = 0.0
            start_ms = int(start_time * 1000)
            # 구간이 길더라도 전체 길이를 초과하지 않도록 위치 조정
            if start_ms < 0:
                start_ms = 0
            # 해당 위치에 음성 구간을 오버레이
            final_audio = final_audio.overlay(segment_audio, position=start_ms)

    # 필요 시 패딩/트리밍으로 길이를 배경 오디오와 동일하게 맞춤
    if len(final_audio) < total_duration_ms:
        silence = AudioSegment.silent(duration=(total_duration_ms - len(final_audio)))
        final_audio = final_audio + silence
    elif len(final_audio) > total_duration_ms:
        final_audio = final_audio[:total_duration_ms]

    # 믹싱된 오디오를 outputs 디렉터리에 저장
    output_dir = paths.outputs_vid_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    final_audio_path = output_dir / "dubbed_audio.wav"
    final_audio.export(str(final_audio_path), format="wav")

    # 원본 영상과 새 오디오를 결합
    output_video_path = output_dir / "dubbed_video.mp4"
    # ffmpeg로 영상의 오디오 트랙을 교체
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_input),
        "-i",
        str(final_audio_path),
        "-c:v",
        "copy",
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-shortest",
        str(output_video_path),
    ]
    subprocess.run(cmd, check=True, timeout=600)  # 10분 타임아웃
    return {
        "output_video": str(output_video_path),
        "output_audio": str(final_audio_path),
    }
