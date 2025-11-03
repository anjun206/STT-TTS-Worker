import subprocess
import shlex
import os
from typing import List, Tuple, Optional, Dict
import re
import tempfile
import math
import glob
import importlib.util

# -------------------- 공용 실행/시간 유틸 --------------------

def run(cmd: str, cwd: Optional[str]=None) -> None:
    print(f"[RUN] {cmd}")
    proc = subprocess.run(shlex.split(cmd), cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(proc.stdout)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}\n{proc.stdout}")

def ffprobe_duration(path: str) -> float:
    cmd = f"ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 {shlex.quote(path)}"
    proc = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr)
    try:
        return float(proc.stdout.strip())
    except:
        return 0.0

# -------------------- 오디오 가공 기본 --------------------

def make_silence(path: str, seconds: float, ar: int = 24000):
    if seconds <= 0.0001:
        # 0초에 가까우면 빈 파일을 만들어 concat 오류 방지
        open(path, "wb").close()
        return
    run(f"ffmpeg -y -f lavfi -i anullsrc=r={ar}:cl=mono -t {seconds:.6f} -ar {ar} -ac 1 {shlex.quote(path)}")

def time_stretch(in_path: str, out_path: str, tempo: float, ar: int = 24000):
    """
    tempo > 1.0 이면 빨라짐(길이 짧아짐), tempo < 1.0 이면 느려짐(길이 길어짐)
    rubberband가 있으면 품질↑. 없으면 atempo 체인.
    """
    # rubberband 우선
    try:
        run(f'ffmpeg -y -i {shlex.quote(in_path)} -af "rubberband=tempo={tempo:.6f}:formant=1" -ar {ar} -ac 1 {shlex.quote(out_path)}')
        return
    except Exception:
        pass
    # fallback: atempo(0.5~2.0 per filter) - 체인 분할
    def split_tempos(t):
        parts = []
        # atempo는 0.5~2.0 범위. 필요한 배속을 이 범위로 분해
        while t < 0.5 or t > 2.0:
            if t < 0.5:
                parts.append(0.5)
                t /= 0.5
            else:
                parts.append(2.0)
                t /= 2.0
        parts.append(t)
        return parts
    filters = ",".join([f"atempo={t:.6f}" for t in split_tempos(tempo)])
    run(f'ffmpeg -y -i {shlex.quote(in_path)} -af "{filters}" -ar {ar} -ac 1 {shlex.quote(out_path)}')

def concat_audio(files: List[str], out_path: str) -> str:
    """
    files: 24k/mono wav들 경로 순서. copy concat.
    """
    if not files:
        raise RuntimeError("concat_audio: no input files")
    list_path = os.path.join(os.path.dirname(out_path), "_concat.txt")
    with open(list_path, "w", encoding="utf-8") as f:
        for p in files:
            f.write(f"file '{os.path.basename(p)}'\n")
    run(f"ffmpeg -y -f concat -safe 0 -i {list_path} -c copy {shlex.quote(out_path)}")
    return out_path

def trim_or_pad_to_duration(in_path: str, out_path: str, target_dur: float, ar: int = 24000, fade_out_ms: int = 30):
    """
    입력을 target_dur로 정확히 맞춤.
    - 짧으면 뒤에 무음 붙임
    - 길면 살짝 페이드아웃하며 트림(클릭 방지)
    """
    cur = ffprobe_duration(in_path)
    if cur <= 0.0:
        # 비정상 파일이면 그냥 타겟 길이의 무음으로 대체
        make_silence(out_path, target_dur, ar=ar)
        return {"padded": target_dur, "trimmed": 0.0}
    if cur < target_dur - 1e-3:
        pad = target_dur - cur
        sil = os.path.join(os.path.dirname(out_path), "_pad.wav")
        make_silence(sil, pad, ar=ar)
        tmp = os.path.join(os.path.dirname(out_path), "_to_concat.wav")
        run(f"ffmpeg -y -i {shlex.quote(in_path)} -ar {ar} -ac 1 {shlex.quote(tmp)}")
        out24 = os.path.join(os.path.dirname(out_path), "_out24.wav")
        concat_audio([tmp, sil], out24)
        run(f"ffmpeg -y -i {out24} -ar {ar} -ac 1 {shlex.quote(out_path)}")
        return {"padded": pad, "trimmed": 0.0}
    elif cur > target_dur + 1e-3:
        # 끝 페이드 후 정확 트림
        st = max(0.0, target_dur - fade_out_ms/1000.0)
        run(
            f'ffmpeg -y -i {shlex.quote(in_path)} '
            f'-af "afade=t=out:st={st:.6f}:d={fade_out_ms/1000.0:.6f},atrim=0:{target_dur:.6f},asetpts=PTS-STARTPTS" '
            f'-ar {ar} -ac 1 {shlex.quote(out_path)}'
        )
        return {"padded": 0.0, "trimmed": cur - target_dur}
    else:
        # 거의 동일: 리샘플/모노만 보장
        run(f"ffmpeg -y -i {shlex.quote(in_path)} -ar {ar} -ac 1 {shlex.quote(out_path)}")
        return {"padded": 0.0, "trimmed": 0.0}

def replace_audio_in_video(video_in: str, audio_in: str, video_out: str) -> None:
    # 오디오 PTS 0부터, AAC 48k, faststart
    run(
        f'ffmpeg -y -i {shlex.quote(video_in)} -i {shlex.quote(audio_in)} '
        f'-map 0:v:0 -map 1:a:0 -c:v copy '
        f'-af "asetpts=PTS-STARTPTS" -c:a aac -ar 48000 -b:a 192k '
        f'-shortest -movflags +faststart {shlex.quote(video_out)}'
    )

# -------------------- (기존) 분석/도움 함수들 --------------------

def detect_silences(wav_path: str, noise_db: str = "-35dB", min_s: float = 0.25):
    cmd = (
        f'ffmpeg -hide_banner -nostats -i {shlex.quote(wav_path)} '
        f'-af silencedetect=noise={noise_db}:d={min_s} -f null -'
    )
    proc = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    txt = proc.stdout
    silences = []
    last = None
    for line in txt.splitlines():
        m1 = re.search(r"silence_start:\s*([0-9.]+)", line)
        m2 = re.search(r"silence_end:\s*([0-9.]+)", line)
        if m1:
            last = float(m1.group(1))
        if m2 and last is not None:
            silences.append((last, float(m2.group(1))))
            last = None
    return silences

def segment_boundaries(duration: float, silences):
    pts = [0.0]
    for s,e in silences:
        pts += [s,e]
    pts.append(duration)
    pts = [max(0.0, p) for p in sorted(set(pts))]
    out = []
    for i in range(len(pts)-1):
        start, end = pts[i], pts[i+1]
        is_sil = any(abs(start - s) < 1e-3 and abs(end - e) < 1e-3 for s,e in silences)
        out.append((start, end, is_sil))
    return out

def cut_wav_segment(src: str, dst: str, start: float, end: float, ar: int = 24000):
    start = max(0.0, float(start)); end = max(start, float(end))
    run(f'ffmpeg -y -ss {start:.6f} -to {end:.6f} -i {shlex.quote(src)} -ar {ar} -ac 1 {shlex.quote(dst)}')

def _nearest_silence_time(target: float, silences, window: float = 0.25):
    """
    target 근처(±window초)에서 가장 가까운 무음 중앙점(mid)을 반환.
    없으면 None.
    """
    best_t, best_d = None, 1e9
    for s,e in silences:
        mid = 0.5*(s+e)
        if (target - window) <= mid <= (target + window):
            d = abs(mid - target)
            if d < best_d:
                best_d, best_t = d, mid
    return best_t

def split_audio_by_targets(raw_wav: str, target_durations: list[float], out_dir: str, prefix: str,
                           search_window: float = 0.25, ar: int = 24000) -> list[str]:
    """
    raw_wav을 target durations(슬롯 길이들)의 누적합 경계에 맞춰 N개로 자름.
    각 경계는 '근처 무음(mid)'로 스냅(없으면 그 시간 그대로).
    반환: 잘린 WAV 경로 리스트 길이 == len(target_durations)
    """
    total = ffprobe_duration(raw_wav)
    targets = [max(0.05, float(d)) for d in target_durations]
    # 경계 후보 (0, t1, t2, ..., total or sum(targets))
    cum = [0.0]
    acc = 0.0
    for d in targets[:-1]:  # 마지막 경계는 total에 맡긴다(언더/오버라도 자투리 흡수)
        acc += d; cum.append(acc)
    # 실제 총합과 raw 길이 차이를 고려
    raw_end = total
    boundaries = [0.0] + [min(max(0.0, t), raw_end) for t in cum[1:]] + [raw_end]

    # 무음에 스냅
    sils = detect_silences(raw_wav, noise_db="-35dB", min_s=0.08)
    snapped = [boundaries[0]]
    for t in boundaries[1:-1]:
        snap = _nearest_silence_time(t, sils, window=search_window)
        snapped.append(snap if snap is not None else t)
    snapped.append(boundaries[-1])

    # 단조성 보정(겹침 제거)
    for i in range(1, len(snapped)):
        if snapped[i] <= snapped[i-1] + 1e-3:
            snapped[i] = snapped[i-1] + 1e-3
    if snapped[-1] < snapped[-2] + 0.02:
        snapped[-1] = snapped[-2] + 0.02

    # 컷
    outs = []
    for i in range(len(targets)):
        s, e = snapped[i], snapped[i+1]
        part = os.path.join(out_dir, f"{prefix}_part_{i:02d}.wav")
        cut_wav_segment(raw_wav, part, s, e, ar=ar)
        outs.append(part)
    return outs

def extract_audio_full(video_in: str, wav_out: str):
    # 원본 오디오를 48k 스테레오 wav로 추출
    run(f"ffmpeg -y -i {shlex.quote(video_in)} -map 0:a:0 -ac 2 -ar 48000 -vn -c:a pcm_s16le {shlex.quote(wav_out)}")

def separate_bgm_vocals(in_wav: str, out_vocals: str, out_bgm: str, model: str = "htdemucs"):
    """
    Demucs 2-stems 분리: vocals / no_vocals
    demucs 미설치 또는 실행 실패 시 즉시 예외를 발생시킨다.
    """
    if importlib.util.find_spec("demucs") is None:
        raise RuntimeError("demucs package not installed; background separation is unavailable")

    outdir = os.path.join(os.path.dirname(out_vocals), "sep")
    try:
        run(
            f"python -m demucs.separate -n {model} --two-stems=vocals -o {shlex.quote(outdir)} {shlex.quote(in_wav)}"
        )
        base = os.path.splitext(os.path.basename(in_wav))[0]
        cand_dir = glob.glob(os.path.join(outdir, model, base))
        if not cand_dir:
            raise RuntimeError("demucs output not found")
        cand_dir = cand_dir[0]
        v = glob.glob(os.path.join(cand_dir, "vocals.wav"))[0]
        nv = glob.glob(os.path.join(cand_dir, "no_vocals.wav"))[0]
        run(f"ffmpeg -y -i {shlex.quote(v)} -ar 48000 -ac 2 {shlex.quote(out_vocals)}")
        run(f"ffmpeg -y -i {shlex.quote(nv)} -ar 48000 -ac 2 {shlex.quote(out_bgm)}")
    except Exception as exc:
        raise RuntimeError("demucs separation failed") from exc

def detect_mean_volume_db(wav_path: str) -> Optional[float]:
    """
    ffmpeg volumedetect를 이용해 평균 음량(dBFS)을 반환. 측정 실패 시 None.
    """
    try:
        proc = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-nostats",
                "-i",
                wav_path,
                "-af",
                "volumedetect",
                "-f",
                "null",
                "-",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:  # pragma: no cover - 환경 이슈
        raise RuntimeError("ffmpeg not found while measuring volume") from exc

    if proc.returncode != 0:
        # volumedetect는 stderr에 기록 후 0으로 종료하지만, 입력이 짧으면 실패할 수 있음
        output = (proc.stdout or "") + (proc.stderr or "")
    else:
        output = proc.stderr or ""

    match = re.search(r"mean_volume:\s*(-?[0-9.]+|-\s*inf)\s*dB", output)
    if not match:
        return None
    value = match.group(1).replace(" ", "")
    if value.lower() in {"-inf", "inf"}:
        return None
    try:
        return float(value)
    except ValueError:
        return None

def apply_gain_db(in_wav: str, out_wav: str, gain_db: float, ar: int = 48000, ac: int = 1):
    """
    지정한 dB 게인을 적용. 양수면 증폭, 음수면 감쇠.
    """
    if abs(gain_db) < 1e-3 and os.path.abspath(in_wav) == os.path.abspath(out_wav):
        # 자가 대입이면서 변경 불필요하면 종료
        return
    run(
        f'ffmpeg -y -i {shlex.quote(in_wav)} -af "volume={gain_db:.4f}dB" '
        f"-ar {ar} -ac {ac} {shlex.quote(out_wav)}"
    )

def mix_bgm_with_tts(bgm_wav: str, tts_wav: str, out_wav: str):
    """
    BGM과 TTS를 동일한 레벨로 단순 합성.
    - 최종 출력: 48k 스테레오
    """
    run(
        'ffmpeg -y -i {bgm} -i {tts} -filter_complex '
        '"[1:a]aformat=channel_layouts=mono,pan=stereo|c0=c0|c1=c0[tts_st];'
        ' [0:a][tts_st]amix=inputs=2:duration=first:dropout_transition=0:normalize=0,'
        'alimiter=limit=0.95" '
        '-ar 48000 -ac 2 {out}'.format(
            bgm=shlex.quote(bgm_wav), tts=shlex.quote(tts_wav), out=shlex.quote(out_wav)
        )
    )

def mask_keep_intervals(in_wav: str, keep: list[tuple[float, float]], out_wav: str, sr: int = 48000, ac: int = 2):
    """
    keep 구간만 원본 레벨 그대로 두고, 그 외는 볼륨 0으로 '침묵화'하여 길이를 보존.
    ffmpeg volume 필터의 enable를 이용해 not(keep)을 0으로 만듦.
    """
    dur = ffprobe_duration(in_wav)
    if dur <= 0.0:
        # 입력이 이상하면 동일 길이 무음
        make_silence(out_wav, 0.0, ar=sr)
        return

    if not keep:
        # 전부 비-스피치: 전체 길이 무음
        run(f"ffmpeg -y -f lavfi -i anullsrc=channel_layout={'stereo' if ac==2 else 'mono'}:sample_rate={sr} -t {dur:.6f} -ar {sr} -ac {ac} {shlex.quote(out_wav)}")
        return

    expr = "+".join([f"between(t,{s:.6f},{e:.6f})" for s, e in keep])
    # keep 외 구간은 볼륨 0 → 타임라인 보존
    run(
        f'ffmpeg -y -i {shlex.quote(in_wav)} '
        f'-af "volume=0:enable=\'not({expr})\'" -ar {sr} -ac {ac} {shlex.quote(out_wav)}'
    )

def mix_bgm_fx_with_tts(bgm_wav: str, fx_wav: str, tts_wav: str, out_wav: str):
    """
    (BGM + 비-스피치 FX)를 먼저 합친 뒤 TTS와 동일 레벨로 섞는다.
    최종 출력은 48k 스테레오.
    """
    run(
        'ffmpeg -y -i {bgm} -i {fx} -i {tts} -filter_complex '
        '"[0:a][1:a]amix=inputs=2:duration=first:dropout_transition=0:normalize=0,'
        'aformat=channel_layouts=stereo[bed];'
        ' [2:a]aformat=channel_layouts=mono,pan=stereo|c0=c0|c1=c0[tts_st];'
        ' [bed][tts_st]amix=inputs=2:duration=first:dropout_transition=0:normalize=0,'
        'alimiter=limit=0.95" '
        '-ar 48000 -ac 2 {out}'.format(
            bgm=shlex.quote(bgm_wav), fx=shlex.quote(fx_wav), tts=shlex.quote(tts_wav), out=shlex.quote(out_wav)
        )
    )
