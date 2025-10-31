# app/pipeline.py
import os
import uuid
import json
import shutil
import shlex
from typing import Optional, List, Dict

from fastapi import UploadFile

from .tts import synthesize
from .utils import (
    run, ffprobe_duration, make_silence, time_stretch,
    trim_or_pad_to_duration, concat_audio, replace_audio_in_video,
    split_audio_by_targets, separate_bgm_vocals, mix_bgm_with_tts,
    extract_audio_full,mask_keep_intervals, mix_bgm_fx_with_tts,
    cut_wav_segment,
)
from .utils_meta import load_meta, save_meta
from .vad import compute_vad_silences, sum_silence_between, complement_intervals, merge_intervals

from typing import Tuple

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")  # base|small|medium|large-v3
TTS_MODEL = os.getenv("TTS_MODEL", "tts_models/multilingual/multi-dataset/xtts_v2")

USE_GPU = os.getenv("USE_GPU", "1") == "1"
_device = "cuda" if USE_GPU else "cpu"
_compute = "float16" if _device == "cuda" else "int8"

_whisper_model = None  # Lazily populated when ASR is requested

EPS = 0.02  # 20ms 허용오차

# ----------------- 공통 유틸 -----------------
def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def _extract_tracks(in_path: str, work: str) -> Tuple[str, str, str, str]:
    """
    전체 오디오(48k) 추출 → 보이스/배경 분리 → 보이스 16k/mono까지 반환
    returns: (full_48k, vocals_48k, bgm_48k, vocals_16k_raw)
    """
    full_48k = os.path.join(work, "audio_full_48k.wav")
    extract_audio_full(in_path, full_48k)

    vocals_48k = os.path.join(work, "vocals_48k.wav")
    bgm_48k    = os.path.join(work, "bgm_48k.wav")
    if os.getenv("SEPARATE_BGM", "1") == "1":
        separate_bgm_vocals(full_48k, vocals_48k, bgm_48k)
    else:
        run(f"ffmpeg -y -i {shlex.quote(full_48k)} -ar 48000 -ac 2 {shlex.quote(vocals_48k)}")
        run(f"ffmpeg -y -f lavfi -i anullsrc=channel_layout=stereo:sample_rate=48000 -t {ffprobe_duration(full_48k):.3f} {shlex.quote(bgm_48k)}")

    vocals_16k_raw = os.path.join(work, "vocals_16k_raw.wav")
    run(f"ffmpeg -y -i {shlex.quote(vocals_48k)} -ac 1 -ar 16000 -c:a pcm_s16le {shlex.quote(vocals_16k_raw)}")
    return full_48k, vocals_48k, bgm_48k, vocals_16k_raw

def _pick_ref(audio_wav_16k: str, out_ref_24k: str):
    # 6초 참조 (24k mono)
    run(f"ffmpeg -y -i {audio_wav_16k} -t 6 -ar 24000 -ac 1 {out_ref_24k}")

def _whisper_transcribe(audio_wav_16k: str):
    global _whisper_model
    if _whisper_model is None:
        try:
            from faster_whisper import WhisperModel  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "ASR requested but faster-whisper is not installed in this environment."
            ) from e
        _whisper_model = WhisperModel(WHISPER_MODEL, device=_device, compute_type=_compute)
    model = _whisper_model
    segments, info = model.transcribe(audio_wav_16k, language=None, vad_filter=True, word_timestamps=False)
    return [{"start": float(s.start), "end": float(s.end), "text": s.text.strip()} for s in segments]


def _ensure_ref_voice(work: str, wav_16k: str, ref_voice: Optional[UploadFile]) -> str:
    ref_path = os.path.join(work, "ref.wav")
    if ref_voice is not None:
        with open(ref_path, "wb") as f:
            shutil.copyfileobj(ref_voice.file, f)
        return ref_path
    _pick_ref(wav_16k, ref_path)
    return ref_path

def synthesize_single_text(
    text: str,
    target_lang: str,
    ref_voice: UploadFile,
    *,
    tmp_root: str = "/app/data/tmp",
) -> Dict[str, str]:
    """
    업로드한 참조 음성을 그대로 사용해 단일 문장을 합성.
    반환: {"job_id": str, "workdir": str, "tts_wav": str, "ref_wav": str}
    """
    if ref_voice is None:
        raise ValueError("ref_voice is required")
    assert target_lang in ("en", "ja", "ko"), "target_lang must be 'en', 'ja', or 'ko'"

    _ensure_dir(tmp_root)
    job_id = uuid.uuid4().hex
    workdir = os.path.join(tmp_root, f"tts_{job_id}")
    _ensure_dir(workdir)

    # 업로드 파일 저장
    original_name = getattr(ref_voice, "filename", None) or "ref_input.wav"
    raw_ref = os.path.join(workdir, os.path.basename(original_name))
    if hasattr(ref_voice.file, "seek"):
        try:
            ref_voice.file.seek(0)
        except Exception:
            pass

    with open(raw_ref, "wb") as f:
        shutil.copyfileobj(ref_voice.file, f)

    out_path = os.path.join(workdir, "tts.wav")
    synthesize(text, raw_ref, language=target_lang, out_path=out_path, model_name=TTS_MODEL)
    return {"job_id": job_id, "workdir": workdir, "tts_wav": out_path, "ref_wav": raw_ref}



def _translate_texts_safe(texts: List[str], src: str, tgt: str) -> List[str]:
    try:
        from .translate import translate_texts  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Translation dependencies are missing; install transformers stack to use this endpoint."
        ) from e
    return translate_texts(texts, src=src, tgt=tgt)

# ----------------- 번역 (자동 수정 없음) -----------------
def translate_stage(segments: List[Dict], src: str, tgt: str, length_mode: str = "off") -> List[Dict]:
    """
    �ڵ� ���� ����/���� ���� ����.
    �Է� ���׸�Ʈ�� ������ǥ(start,end)�� �״�� ������ ���� �ؽ�Ʈ�� ä��.
    ��ȯ: [{"start": float, "end": float, "text": str}, ...]
    """
    texts = [s["text"] for s in segments]
    base = _translate_texts_safe(texts, src=src, tgt=tgt)
    outs: List[Dict] = []
    for s, t in zip(segments, base):
        outs.append({"start": s["start"], "end": s["end"], "text": t})
    return outs



# ----------------- 1차 TTS: 길이 측정(Probe) -----------------
async def tts_probe_stage(job_id: str, target_lang: str, ref_voice: Optional[UploadFile]):
    """
    세그먼트별로 TTS를 '원문 그대로' 합성하여 실제 길이를 측정하고,
    slot(=end-start)과 비교해 over/less/fit 및 초 차이(delta)를 기록.
    오디오는 결합하지 않음. 리포트만 저장/반환.
    """
    assert target_lang in ("en", "ja", "ko")
    work = os.path.join("/app/data", job_id)
    meta = load_meta(work)
    assert "translations" in meta, "No translations"
    assert "wav_16k" in meta and "input" in meta, "Invalid meta"

    trs: List[Dict] = meta["translations"]
    ref = _ensure_ref_voice(work, meta["wav_16k"], ref_voice)

    report = []
    for i, tr in enumerate(trs):
        start = float(tr["start"]); end = float(tr["end"])
        slot = max(0.05, end - start)

        raw = os.path.join(work, f"probe_{i:04d}_raw.wav")
        synthesize(tr["text"], ref, language=target_lang, out_path=raw, model_name=TTS_MODEL)
        raw_dur = ffprobe_duration(raw)

        delta = raw_dur - slot
        if delta > EPS:
            status = "over"
        elif delta < -EPS:
            status = "less"
        else:
            status = "fit"

        report.append({
            "i": i,
            "start": start,
            "end": end,
            "slot_dur": slot,
            "raw_dur": raw_dur,
            "delta": delta,
            "status": status
        })

    meta["duration_report"] = report
    save_meta(work, meta)
    return {"workdir": work, "duration_report": report}

# ----------------- 2차 TTS: 최종 보정/결합(Finalize) -----------------
async def tts_finalize_stage(job_id: str, target_lang: str, ref_voice: Optional[UploadFile]):
    assert target_lang in ("en", "ja", "ko")
    work = os.path.join("/app/data", job_id)
    meta = load_meta(work)
    assert "translations" in meta and "wav_16k" in meta and "input" in meta

    trs: List[Dict] = meta["translations"]
    ref = _ensure_ref_voice(work, meta["wav_16k"], ref_voice)
    layout = meta.get("merge_layouts", {})  # new_index 기준

    parts: List[str] = []
    final_report: List[Dict] = []

    # 🔹 각 세그먼트의 '오른쪽 절대 갭' 미리 계산
    right_gaps: List[float] = []
    for i in range(len(trs)):
        if i < len(trs) - 1:
            rg = max(0.0, float(trs[i+1]["start"]) - float(trs[i]["end"]))
        else:
            rg = 0.0
        right_gaps.append(rg)

    # 리딩 갭
    lead = max(0.0, float(trs[0]["start"]))
    if lead > 0.0001:
        lead_wav = os.path.join(work, "lead.wav")
        make_silence(lead_wav, lead, ar=24000)
        parts.append(lead_wav)

    for i, tr in enumerate(trs):
        start = float(tr["start"]); end = float(tr["end"])
        slot = max(0.05, end - start)

        # 외부 갭(세그 i와 i+1 사이)에 실제로 삽입할 길이(빌림 적용 후)
        post_gap: Optional[float] = None

        # 이 번역 세그가 머지된 것인지 확인
        lay = layout.get(i)
        if lay and len(lay.get("slots", [])) > 1:
            # 🔹 머지된 세그먼트: 텍스트 1번 합성 → 원본 슬롯 길이에 따라 '스마트 컷'
            raw_all = os.path.join(work, f"final_{i:04d}_raw_all.wav")
            synthesize(tr["text"], ref, language=target_lang, out_path=raw_all, model_name=TTS_MODEL)
            slot_durs = [max(0.05, float(s["end"]) - float(s["start"])) for s in lay["slots"]]
            inner_gaps = [max(0.0, float(g)) for g in lay.get("gaps", [])]

            # 무음 스냅 컷
            chunks = split_audio_by_targets(raw_all, slot_durs, work, f"final_{i:04d}")
            # 각 슬롯별 규칙 적용(오버=배속, 레스=패드)
            for j, ch in enumerate(chunks):
                ch_dur = ffprobe_duration(ch); tgt = slot_durs[j]
                outp = os.path.join(work, f"final_{i:04d}_slot_{j:02d}.wav")
                if ch_dur > tgt + EPS:
                    tempo = ch_dur / tgt
                    tmp = outp.replace(".wav", "_tempo.wav")
                    time_stretch(ch, tmp, tempo=tempo, ar=24000)
                    info = trim_or_pad_to_duration(tmp, outp, tgt, ar=24000)
                    final_report.append({"i": i, "sub": j, "mode": "speedup",
                                         "tempo": tempo, "raw_dur": ch_dur,
                                         "slot_dur": tgt, "padded": info["padded"], "trimmed": info["trimmed"]})
                else:
                    info = trim_or_pad_to_duration(ch, outp, tgt, ar=24000)
                    final_report.append({"i": i, "sub": j, "mode": "pad",
                                         "raw_dur": ch_dur, "slot_dur": tgt,
                                         "padded": info["padded"], "trimmed": info["trimmed"]})
                parts.append(outp)

                # 내부 갭 삽입
                if j < len(inner_gaps):
                    gap = inner_gaps[j]
                    if gap > 0.0001:
                        g = os.path.join(work, f"final_{i:04d}_gap_{j:02d}.wav")
                        make_silence(g, gap, ar=24000)
                        parts.append(g)

            # 머지 케이스는 외부 갭을 빌리지 않고 기본값 사용
            if i < len(trs) - 1:
                post_gap = right_gaps[i]

        else:
            # 🔸 일반(머지 아님 or 단일 슬롯) 처리: 오른쪽 갭에서 시간 '빌려' slot 확대
            raw = os.path.join(work, f"final_{i:04d}_raw.wav")
            synthesize(tr["text"], ref, language=target_lang, out_path=raw, model_name=TTS_MODEL)
            raw_dur = ffprobe_duration(raw)

            need   = max(0.0, raw_dur - slot)         # 모자란 시간
            borrow = min(need, right_gaps[i])          # 오른쪽 갭에서 빌릴 수 있는 만큼
            slot_used = slot + borrow                  # 실사용 슬롯
            gap_after = max(0.0, right_gaps[i] - borrow)  # 남겨둘 외부 갭
            post_gap = gap_after

            fit = os.path.join(work, f"final_{i:04d}_slot.wav")
            if raw_dur > slot_used + EPS:
                tempo = raw_dur / slot_used
                tmp = fit.replace(".wav", "_tempo.wav")
                time_stretch(raw, tmp, tempo=tempo, ar=24000)
                info = trim_or_pad_to_duration(tmp, fit, slot_used, ar=24000)
                final_report.append({
                    "i": i, "mode": "speedup+borrow", "tempo": tempo,
                    "borrowed": borrow, "raw_dur": raw_dur,
                    "slot": slot, "slot_used": slot_used,
                    "padded": info["padded"], "trimmed": info["trimmed"]
                })
            else:
                info = trim_or_pad_to_duration(raw, fit, slot_used, ar=24000)
                final_report.append({
                    "i": i, "mode": "pad/fit+borrow",
                    "borrowed": borrow, "raw_dur": raw_dur,
                    "slot": slot, "slot_used": slot_used,
                    "padded": info["padded"], "trimmed": info["trimmed"]
                })
            parts.append(fit)

        # 🔹 외부(세그 간) 절대 갭 삽입: 빌림 반영한 post_gap 사용
        if i < len(trs) - 1:
            gap = float(post_gap if post_gap is not None else right_gaps[i])
            if gap > 0.0001:
                g = os.path.join(work, f"gap_{i:04d}.wav")
                make_silence(g, gap, ar=24000)
                parts.append(g)

    # 24k concat → 48k
    out24 = os.path.join(work, "dubbed_24k.wav")
    concat_audio(parts, out24)
    dubbed = os.path.join(work, "dubbed.wav")
    run(f"ffmpeg -y -i {out24} -ar 48000 -ac 1 {dubbed}")

    meta["dubbed_wav"] = dubbed
    meta["final_report"] = final_report
    save_meta(work, meta)
    return {"workdir": work, "dubbed_wav": dubbed, "final_report": final_report}

# ----------------- ASR/번역/원샷 -----------------
async def asr_only(file: UploadFile) -> Dict:
    job_id = str(uuid.uuid4())[:8]
    work = os.path.join("/app/data", job_id); _ensure_dir(work)
    in_path = os.path.join(work, file.filename)
    with open(in_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # 1) 트랙 추출
    full_48k, vocals_48k, bgm_48k, vocals_16k_raw = _extract_tracks(in_path, work)
    total = ffprobe_duration(full_48k)

    # 2) (참고용) VAD 침묵 구간: gap 계산에만 사용
    silences = compute_vad_silences(
        vocals_16k_raw,
        aggressiveness=int(os.getenv("VAD_AGGR", "3")),
        frame_ms=int(os.getenv("VAD_FRAME_MS", "30")),
    )

    # 3) STT는 raw 보이스(=사람말+울음 포함)에서 직접 돌리기보다,
    #    일단 raw 보이스 16k 그대로 돌립니다. (울음 구간은 대부분 비문/공백)
    segments = _whisper_transcribe(vocals_16k_raw)

    # 4) STT 세그 기반 speech 구간(여유 margin 포함) 산출
    margin = float(os.getenv("STT_INTERVAL_MARGIN", "0.10"))  # ±100ms
    stt_intervals = merge_intervals([
        (max(0.0, float(s["start"]) - margin), min(float(total), float(s["end"]) + margin))
        for s in segments if float(s["end"]) > float(s["start"])
    ])

    # 5) 사람말 전용/FX 트랙 만들기 (타임라인 보존)
    speech_only_48k = os.path.join(work, "speech_only_48k.wav")
    vocals_fx_48k   = os.path.join(work, "vocals_fx_48k.wav")
    mask_keep_intervals(vocals_48k, stt_intervals, speech_only_48k, sr=48000, ac=2)
    nonspeech_intervals = complement_intervals(stt_intervals, total)
    mask_keep_intervals(vocals_48k, nonspeech_intervals, vocals_fx_48k, sr=48000, ac=2)

    # 6) STT/후속 처리는 "사람말만 담긴" 트랙에서 진행 (타임라인 동일)
    wav_16k = os.path.join(work, "speech_16k.wav")
    run(f"ffmpeg -y -i {shlex.quote(speech_only_48k)} -ac 1 -ar 16000 -c:a pcm_s16le {shlex.quote(wav_16k)}")

    # 필요시, segments를 speech_only에서 재추출하고 싶다면 아래로 대체:
    # segments = _whisper_transcribe(wav_16k)

    # 7) gap 기록(VAD 기준; 원본 타임라인 유지)
    for i in range(len(segments)):
        if i < len(segments) - 1:
            st = float(segments[i]["end"]); en = float(segments[i+1]["start"])
            segments[i]["gap_after_vad"] = sum_silence_between(silences, st, en)
            segments[i]["gap_after"] = max(0.0, en - st)
        else:
            segments[i]["gap_after_vad"] = 0.0
            segments[i]["gap_after"] = 0.0

    meta = {
        "job_id": job_id,
        "workdir": work,
        "input": in_path,
        "audio_full_48k": full_48k,
        "vocals_48k": vocals_48k,
        "bgm_48k": bgm_48k,
        "speech_only_48k": speech_only_48k,   # ✅ 사람말만
        "vocals_fx_48k":  vocals_fx_48k,      # ✅ 동물/환호 등 비-스피치
        "wav_16k": wav_16k,                   # ✅ 이후 파이프라인 입력
        "orig_duration": total,
        "segments": segments,
        "silences": silences,
        "speech_intervals_stt": stt_intervals,
        "nonspeech_intervals_stt": nonspeech_intervals,
    }
    save_meta(work, meta)
    return meta

def mux_stage(job_id: str) -> str:
    work = os.path.join("/app/data", job_id)
    meta = load_meta(work)
    assert "input" in meta and "dubbed_wav" in meta

    out_video = os.path.join(work, "output.mp4")

    # 1) BGM + 비-스피치 FX + TTS 믹스(ducking)
    mix = os.path.join(work, "final_mix.wav")
    fx = meta.get("vocals_fx_48k", meta.get("bgm_48k"))  # 없으면 bgm로 폴백
    mix_bgm_fx_with_tts(meta["bgm_48k"], fx, meta["dubbed_wav"], mix)

    # 2) 비디오와 합치기
    replace_audio_in_video(meta["input"], mix, out_video)
    meta["final_mix"] = mix
    meta["output"] = out_video
    save_meta(work, meta)
    return out_video


# (선택) 원샷 dub: 번역 자동수정 없이, 최종 규칙으로 바로 처리
def dub(video_in: str, target_lang: str, ref_wav: Optional[str] = None) -> Dict:
    job_id = str(uuid.uuid4())[:8]
    work = os.path.join("/app/data", job_id); _ensure_dir(work)

    # 1) 오디오 추출 (+분리)
    wav_16k = os.path.join(work, "audio_16k.wav")
    full_48k, vocals_48k, bgm_48k = _extract_audio_16k_mono(video_in, wav_16k)  # ★ 받기

    # 2) 레퍼런스
    ref_path = ref_wav or os.path.join(work, "ref.wav")
    if not ref_wav:
        _pick_ref(wav_16k, ref_path)

    # 3) ASR
    segments = _whisper_transcribe(wav_16k)
    if not segments:
        raise RuntimeError("No speech detected.")

    # 4) MT (자동 수정 없음)
    base = _translate_texts_safe([s["text"] for s in segments], src="ko", tgt=target_lang)
    translations = [{"start": s["start"], "end": s["end"], "text": t} for s, t in zip(segments, base)]

    # 5) 합성/보정/결합
    parts: List[str] = []
    lead = max(0.0, float(translations[0]["start"]))
    if lead > 0.0001:
        lead_wav = os.path.join(work, "lead.wav")
        make_silence(lead_wav, lead, ar=24000)
        parts.append(lead_wav)

    final_report: List[Dict] = []
    for i, tr in enumerate(translations):
        start = float(tr["start"]); end = float(tr["end"]); slot = max(0.05, end - start)
        raw = os.path.join(work, f"seg_{i:04d}_raw.wav")
        synthesize(tr["text"], ref_path, language=target_lang, out_path=raw, model_name=TTS_MODEL)
        raw_dur = ffprobe_duration(raw)

        fit = os.path.join(work, f"seg_{i:04d}_slot.wav")
        if raw_dur > slot + EPS:
            tempo = raw_dur / slot
            tmp = fit.replace(".wav", "_tempo.wav")
            time_stretch(raw, tmp, tempo=tempo, ar=24000)
            info = trim_or_pad_to_duration(tmp, fit, slot, ar=24000)
            final_report.append({"i": i, "mode": "speedup", "tempo": tempo,
                                 "raw_dur": raw_dur, "slot_dur": slot,
                                 "padded": info["padded"], "trimmed": info["trimmed"]})
        else:
            info = trim_or_pad_to_duration(raw, fit, slot, ar=24000)
            final_report.append({"i": i, "mode": "pad",
                                 "raw_dur": raw_dur, "slot_dur": slot,
                                 "padded": info["padded"], "trimmed": info["trimmed"]})
        parts.append(fit)

        if i < len(translations) - 1:
            next_start = float(translations[i+1]["start"])
            gap = max(0.0, next_start - end)
            if gap > 0.0001:
                g = os.path.join(work, f"gap_{i:04d}.wav")
                make_silence(g, gap, ar=24000)
                parts.append(g)

    out24 = os.path.join(work, "dubbed_24k.wav")
    concat_audio(parts, out24)
    dubbed_wav = os.path.join(work, "dubbed.wav")
    run(f"ffmpeg -y -i {out24} -ar 48000 -ac 1 {dubbed_wav}")

    # ★★★ BGM와 믹스 후 비디오에 입히기
    mix = os.path.join(work, "final_mix.wav")
    mix_bgm_with_tts(bgm_48k, dubbed_wav, mix)

    out_video = os.path.join(work, "output.mp4")
    replace_audio_in_video(video_in, mix, out_video)

    meta = {
        "job_id": job_id,
        "target_lang": target_lang,
        "segments": segments,
        "translations": translations,
        "workdir": work,
        "input": video_in,
        "audio_full_48k": full_48k,  # (정보 보존)
        "vocals_48k": vocals_48k,
        "bgm_48k": bgm_48k,
        "wav_16k": wav_16k,
        "dubbed_wav": dubbed_wav,
        "final_mix": mix,            # (정보 보존)
        "final_report": final_report,
        "output": out_video
    }
    with open(os.path.join(work, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return meta

def _apply_merges(segments: List[Dict], merges: List[Tuple[int,int]]) -> Tuple[List[Dict], List[Dict]]:
    """
    merges: [(start_idx, end_idx), ...] 0-based, inclusive, 오름차순/비겹침/연속 범위만 허용
    반환: (merged_segments, merge_map)
      - merged_segments: [{"start":..,"end":..,"text":..}, ...]
      - merge_map: [{"new_index": k, "from": [i..j]} ...]
    """
    n = len(segments)
    if n == 0:
        return [], []

    # 정규화 & 검증
    norm = []
    for s, e in merges or []:
        s, e = int(s), int(e)
        if not (0 <= s <= e < n):
            raise ValueError(f"merge range out of bounds: ({s},{e}) with n={n}")
        norm.append((s, e))
    norm.sort(key=lambda x: x[0])

    # 겹치거나 역순/중첩 금지, 반드시 오름차순 비중첩
    for i in range(1, len(norm)):
        prev = norm[i-1]; cur = norm[i]
        if cur[0] <= prev[1]:
            raise ValueError(f"overlapping merges: {prev} and {cur}")

    merged = []
    mapping = []
    cur_i = 0
    new_idx = 0

    def _append_original(idx):
        nonlocal new_idx
        merged.append({
            "start": float(segments[idx]["start"]),
            "end": float(segments[idx]["end"]),
            "text": segments[idx]["text"],
        })
        mapping.append({"new_index": new_idx, "from": [idx]})
        new_idx += 1

    def _append_merged(a, b):
        nonlocal new_idx
        start = float(segments[a]["start"])
        end   = float(segments[b]["end"])
        # 텍스트는 공백 하나로 연결
        text = " ".join(segments[k]["text"].strip() for k in range(a, b+1) if segments[k]["text"].strip())
        merged.append({"start": start, "end": end, "text": text})
        mapping.append({"new_index": new_idx, "from": list(range(a, b+1))})
        new_idx += 1

    for rng in norm:
        a, b = rng
        # 병합 구간 전의 원본 세그먼트들 추가
        while cur_i < a:
            _append_original(cur_i)
            cur_i += 1
        # 병합 구간 추가
        _append_merged(a, b)
        cur_i = b + 1

    # 남은 꼬리 구간 추가
    while cur_i < n:
        _append_original(cur_i)
        cur_i += 1

    return merged, mapping

def merge_segments_stage(job_id: str, merges: Optional[List[List[int]]] = None):
    work = os.path.join("/app/data", job_id)
    meta = load_meta(work)
    if "segments" not in meta:
        raise RuntimeError("No ASR segments to merge")

    src = meta["segments"]
    all_sil = meta.get("silences", [])
    plan = merges if merges is not None else meta.get("merge_plan")
    if not plan:
        return {"segments": src, "merge_map": [], "note": "no merges applied"}

    # 정규화 & 정렬
    rngs = [(int(a), int(b)) for a,b in plan]
    rngs.sort(key=lambda x: x[0])
    for i in range(1, len(rngs)):
        if rngs[i][0] <= rngs[i-1][1]:
            raise ValueError(f"overlapping merges: {rngs[i-1]} and {rngs[i]}")

    merged = []
    merge_map = []
    merge_layouts = {}  # new_index -> {"from":[...], "slots":[{start,end}], "gaps":[...]}
    cur = 0; new_idx = 0

    def append_original(k):
        nonlocal new_idx
        merged.append({"start": src[k]["start"], "end": src[k]["end"], "text": src[k]["text"]})
        merge_map.append({"new_index": new_idx, "from": [k]})
        # 단일 슬롯(머지 아님)도 레이아웃을 둬두면 일관 처리 쉬움
        merge_layouts[new_idx] = {
            "from": [k],
            "slots": [{"start": src[k]["start"], "end": src[k]["end"]}],
            "gaps":  []
        }
        new_idx += 1

    def append_merged(a,b):
        nonlocal new_idx
        start = float(src[a]["start"]); end = float(src[b]["end"])
        text = " ".join(s["text"].strip() for s in src[a:b+1] if s["text"].strip())
        merged.append({"start": start, "end": end, "text": text})
        merge_map.append({"new_index": new_idx, "from": list(range(a,b+1))})
        slots = [{"start": src[k]["start"], "end": src[k]["end"]} for k in range(a,b+1)]
        # 🔹 내부 갭: VAD 침묵 합으로 계산, 없으면 STT 차이 fallback
        gaps  = []
        for k in range(a, b):
            st = float(src[k]["end"]); en = float(src[k+1]["start"])
            vad_gap = sum_silence_between(meta.get("silences", []), st, en)
            if vad_gap <= 0.0:
                vad_gap = max(0.0, en - st)
            gaps.append(vad_gap)
        merge_layouts[new_idx] = {"from": list(range(a,b+1)), "slots": slots, "gaps": gaps}
        new_idx += 1

    for a,b in rngs:
        while cur < a:
            append_original(cur)
            cur += 1
        append_merged(a,b)
        cur = b+1
    while cur < len(src):
        append_original(cur)
        cur += 1

    # 백업 & 기록
    if "segments_backup" not in meta:
        meta["segments_backup"] = meta["segments"]
    meta["segments"] = merged
    meta["merge_history"] = meta.get("merge_history", []) + [{"plan": plan, "merge_map": merge_map}]
    meta["merge_layouts"] = merge_layouts

    # 이전 산물 무효화
    for k in ("translations","duration_report","final_report","dubbed_wav","output"):
        meta.pop(k, None)

    save_meta(work, meta)
    return {"segments": merged, "merge_map": merge_map}

# ----------------- Voice Sample: 무음 제거 후 음성만 연결 -----------------
def build_voice_sample_stage(job_id: str, out_sr: int = 48000):
    """
    ASR 세그먼트 기준으로 스피치 구간만 잘라 순차적으로 연결한 WAV 생성.
    - 입력: /asr 단계가 완료된 job_id가 있어야 함 (segments, speech_only_48k 필요)
    - 출력: voice_sample_24k.wav (기본)

    반환: {"workdir": str, "voice_sample_wav": str, "parts": [str]}
    """
    work = os.path.join("/app/data", job_id)
    meta = load_meta(work)
    if "segments" not in meta:
        raise RuntimeError("No ASR segments for this job")

    # 스피치만 남긴 48k 트랙 (타임라인 보존됨)
    src_wav = meta.get("speech_only_48k") or meta.get("vocals_48k")
    if not src_wav or not os.path.exists(src_wav):
        # 폴백: 전체 오디오에서 잘라내기
        src_wav = meta.get("audio_full_48k")
    if not src_wav or not os.path.exists(src_wav):
        raise RuntimeError("No source wav to cut from")

    segs: List[Dict] = meta["segments"]
    min_gap = float(os.getenv("VOICE_SAMPLE_MIN_GAP", "0.2"))
    max_gap = float(os.getenv("VOICE_SAMPLE_MAX_GAP", "1.5"))
    parts: List[str] = []
    for i, s in enumerate(segs):
        st = float(s.get("start", 0.0)); en = float(s.get("end", 0.0))
        if en <= st + 1e-3:
            continue
        part = os.path.join(work, f"sample_part_{i:04d}.wav")
        # 24k/mono로 컷하여 저장
        cut_wav_segment(src_wav, part, st, en, ar=out_sr)
        parts.append(part)

        if i < len(segs) - 1:
            raw_gap = float(s.get("gap_after_vad") or s.get("gap_after") or 0.0)
            gap_sec = min(max(raw_gap, min_gap), max_gap) if raw_gap > 0 else min_gap
            if gap_sec > 1e-3:
                gap_wav = os.path.join(work, f"sample_gap_{i:04d}.wav")
                make_silence(gap_wav, gap_sec, ar=out_sr)
                parts.append(gap_wav)

    if not parts:
        raise RuntimeError("No non-empty segments to build sample")

    out24 = os.path.join(work, "voice_sample_24k.wav" if out_sr == 24000 else f"voice_sample_{out_sr//1000}k.wav")
    concat_audio(parts, out24)

    meta["voice_sample_wav"] = out24
    save_meta(work, meta)
    return {"workdir": work, "voice_sample_wav": out24, "parts": parts}
