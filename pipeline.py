"""
Core AI dubbing pipeline.

Steps
-----
1. extract_audio        - ffmpeg: video -> small mono mp3 for transcription
2. transcribe           - ElevenLabs Scribe: auto language detection + speaker diarization + word timestamps
3. build_segments       - group words into timed, single-speaker sentences (knows when each person starts/stops)
4. detect_genders       - pitch (F0) analysis per speaker -> Male / Female
5. translate_segments   - deep-translator (Google) -> target language
6. assign_voices        - one unique ElevenLabs voice per speaker, matched to gender, stays fixed
7. build_dub_track      - ElevenLabs TTS per segment, fitted into the original timing, placed at original start
8. (optional) separate_background - Demucs, keeps music / effects
9. (optional) lipsync   - Wav2Lip in chunks, cut at silences
10. mux_video           - ffmpeg: final video + dubbed audio
"""
from __future__ import annotations

import hashlib
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

Progress = Callable[[float, str], None]


def _noop(p: float, m: str) -> None:  # default progress callback
    pass


# --------------------------------------------------------------------------- #
# Languages
# --------------------------------------------------------------------------- #
# name -> (ISO 639-1 code, code used by Google Translate)
LANGUAGES: Dict[str, Tuple[str, str]] = {
    "English": ("en", "en"),
    "Tamil": ("ta", "ta"),
    "Hindi": ("hi", "hi"),
    "Malayalam": ("ml", "ml"),
    "Telugu": ("te", "te"),
    "Kannada": ("kn", "kn"),
    "Bengali": ("bn", "bn"),
    "Marathi": ("mr", "mr"),
    "Gujarati": ("gu", "gu"),
    "Punjabi": ("pa", "pa"),
    "Urdu": ("ur", "ur"),
    "Nepali": ("ne", "ne"),
    "Arabic": ("ar", "ar"),
    "Spanish": ("es", "es"),
    "French": ("fr", "fr"),
    "German": ("de", "de"),
    "Italian": ("it", "it"),
    "Portuguese": ("pt", "pt"),
    "Russian": ("ru", "ru"),
    "Japanese": ("ja", "ja"),
    "Korean": ("ko", "ko"),
    "Chinese": ("zh", "zh-CN"),
    "Indonesian": ("id", "id"),
    "Turkish": ("tr", "tr"),
    "Thai": ("th", "th"),
    "Vietnamese": ("vi", "vi"),
}

ISO3_TO_1 = {
    "eng": "en", "tam": "ta", "hin": "hi", "mal": "ml", "tel": "te", "kan": "kn",
    "ben": "bn", "mar": "mr", "guj": "gu", "pan": "pa", "urd": "ur", "nep": "ne",
    "ara": "ar", "spa": "es", "fra": "fr", "fre": "fr", "deu": "de", "ger": "de",
    "ita": "it", "por": "pt", "rus": "ru", "jpn": "ja", "kor": "ko", "zho": "zh",
    "chi": "zh", "ind": "id", "tur": "tr", "tha": "th", "vie": "vi",
}

# Languages covered by eleven_multilingual_v2 (the rest of ours need eleven_v3).
V2_LANGS = {"en", "hi", "ta", "ar", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh", "id", "tr"}


def to_iso1(code: Optional[str]) -> str:
    if not code:
        return "unknown"
    code = code.lower()
    return ISO3_TO_1.get(code, code[:2] if len(code) > 3 else code)


def pick_tts_model(lang_code: str, choice: str = "Auto") -> str:
    if choice != "Auto":
        return choice
    return "eleven_multilingual_v2" if lang_code in V2_LANGS else "eleven_v3"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def run(cmd: list, cwd: Optional[str] = None) -> str:
    p = subprocess.run([str(c) for c in cmd], cwd=cwd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(map(str, cmd))}\n{p.stderr[-2500:]}")
    return p.stdout


def media_duration(path: str) -> float:
    out = run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
               "-of", "default=nw=1:nk=1", path])
    return float(out.strip())


def probe_video(path: str) -> Tuple[int, int, float]:
    out = run(["ffprobe", "-v", "error", "-select_streams", "v:0",
               "-show_entries", "stream=width,height,r_frame_rate",
               "-of", "csv=p=0", path]).strip()
    w, h, rate = out.split(",")[:3]
    num, den = rate.split("/") if "/" in rate else (rate, "1")
    fps = float(num) / float(den or 1)
    return int(w), int(h), fps


def fmt_srt_time(t: float) -> str:
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def segments_to_srt(segs: List[dict], key: str = "translated") -> str:
    lines = []
    for i, s in enumerate(segs, 1):
        lines += [str(i), f"{fmt_srt_time(s['start'])} --> {fmt_srt_time(s['end'])}", s.get(key, ""), ""]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 1. Audio extraction
# --------------------------------------------------------------------------- #
def extract_audio(video: str, workdir: Path) -> str:
    out = str(workdir / "source_audio.mp3")
    if not Path(out).exists():
        run(["ffmpeg", "-y", "-i", video, "-vn", "-ac", "1", "-ar", "16000", "-b:a", "48k", out])
    return out


# --------------------------------------------------------------------------- #
# 2. Transcription + diarization (ElevenLabs Scribe)
# --------------------------------------------------------------------------- #
def transcribe(api_key: str, audio_path: str, num_speakers: Optional[int] = None,
               model_id: str = "scribe_v1") -> dict:
    """Returns {"language": iso1, "words": [{text,start,end,type,speaker_id}]}"""
    from elevenlabs.client import ElevenLabs

    client = ElevenLabs(api_key=api_key, timeout=3600)
    kwargs = dict(model_id=model_id, diarize=True, tag_audio_events=False,
                  timestamps_granularity="word")
    if num_speakers:
        kwargs["num_speakers"] = int(num_speakers)
    with open(audio_path, "rb") as f:
        res = client.speech_to_text.convert(file=f, **kwargs)  # language auto-detected

    words = []
    for w in res.words or []:
        words.append({
            "text": w.text,
            "start": getattr(w, "start", None),
            "end": getattr(w, "end", None),
            "type": getattr(w, "type", "word"),
            "speaker_id": getattr(w, "speaker_id", None) or "speaker_0",
        })
    return {"language": to_iso1(getattr(res, "language_code", None)), "words": words}


# --------------------------------------------------------------------------- #
# 3. Segmentation: who speaks, when it starts and stops
# --------------------------------------------------------------------------- #
SENT_END = (".", "?", "!", "।", "॥", "…", "。", "？", "！")


def build_segments(words: List[dict], max_len: float = 15.0, gap: float = 0.8) -> List[dict]:
    segs: List[dict] = []
    cur: Optional[dict] = None

    def close():
        nonlocal cur
        if cur and cur["text"].strip():
            cur["text"] = re.sub(r"\s+", " ", cur["text"]).strip()
            segs.append(cur)
        cur = None

    for w in words:
        t = w.get("type", "word")
        if t == "audio_event":
            continue
        if t == "spacing":
            if cur is not None:
                cur["text"] += w["text"]
            continue
        if w.get("start") is None or w.get("end") is None:
            continue
        spk, start, end = w["speaker_id"], float(w["start"]), float(w["end"])
        if cur and (spk != cur["speaker"] or start - cur["end"] > gap or end - cur["start"] > max_len):
            close()
        if cur is None:
            cur = {"start": start, "end": end, "speaker": spk, "text": ""}
        cur["text"] += w["text"]
        cur["end"] = end
        if w["text"].rstrip().endswith(SENT_END) and cur["end"] - cur["start"] >= 1.5:
            close()
    close()
    return segs


# --------------------------------------------------------------------------- #
# 4. Gender detection (pitch based, per speaker)
# --------------------------------------------------------------------------- #
def detect_genders(audio_path: str, segs: List[dict], threshold_hz: float = 165.0,
                   progress: Progress = _noop) -> List[dict]:
    """Returns one row per speaker (in order of first appearance)."""
    import librosa

    y, sr = librosa.load(audio_path, sr=16000, mono=True)
    order: List[str] = []
    by: Dict[str, List[dict]] = {}
    for s in segs:
        if s["speaker"] not in by:
            order.append(s["speaker"])
        by.setdefault(s["speaker"], []).append(s)

    rows = []
    for n, spk in enumerate(order):
        progress(n / max(len(order), 1), f"Analysing voice of {spk}")
        ss = sorted(by[spk], key=lambda s: s["end"] - s["start"], reverse=True)
        speaking = sum(s["end"] - s["start"] for s in ss)
        f0s, used = [], 0.0
        for s in ss:
            if used >= 40:
                break
            clip = y[int(s["start"] * sr): int(min(s["end"], s["start"] + 10) * sr)]
            if len(clip) < sr * 0.6:
                continue
            f0, _, _ = librosa.pyin(clip, fmin=60, fmax=400, sr=sr, frame_length=2048)
            f0 = f0[~np.isnan(f0)]
            if len(f0):
                f0s.append(f0)
            used += len(clip) / sr
        arr = np.concatenate(f0s) if f0s else np.array([])
        med = float(np.median(arr)) if len(arr) > 20 else 0.0
        gender = "Female" if med >= threshold_hz else "Male"
        rows.append({"speaker": spk, "gender": gender, "median_f0_hz": round(med, 1),
                     "speaking_seconds": round(speaking, 1), "voice_override": ""})
    return rows


# --------------------------------------------------------------------------- #
# 5. Translation
# --------------------------------------------------------------------------- #
def translate_segments(segs: List[dict], target_name: str, source_iso: str,
                       progress: Progress = _noop) -> List[dict]:
    from deep_translator import GoogleTranslator

    tgt_iso, tgt_g = LANGUAGES[target_name]
    out = [dict(s) for s in segs]
    if source_iso == tgt_iso:
        for s in out:
            s["translated"] = s["text"]
        return out

    tr = GoogleTranslator(source="auto", target=tgt_g)
    B = 20
    for i in range(0, len(out), B):
        batch = [s["text"] for s in out[i:i + B]]
        res = None
        for attempt in range(3):
            try:
                res = tr.translate_batch(batch)
                break
            except Exception:
                time.sleep(1.5 * (attempt + 1))
        if res is None:  # fall back to one by one
            res = []
            for t in batch:
                try:
                    res.append(tr.translate(t))
                except Exception:
                    res.append(t)
        for s, r in zip(out[i:i + B], res):
            s["translated"] = r or s["text"]
        progress(min(1.0, (i + B) / len(out)), f"Translating {min(i + B, len(out))}/{len(out)}")
    return out


# --------------------------------------------------------------------------- #
# 6. Voices (unique per speaker, fixed for the whole video)
# --------------------------------------------------------------------------- #
DEFAULT_MALE_VOICES = [
    "pNInz6obpgDQGcFmaJgB | Adam",
    "ErXwobaYiN019PkySvjV | Antoni",
    "TxGEqnHWrfWFTfGW9XjX | Josh",
    "VR6AewLTigWG4xSOukaG | Arnold",
    "yoZ06aMxZJJ28mfd3POQ | Sam",
]
DEFAULT_FEMALE_VOICES = [
    "21m00Tcm4TlvDq8ikWAM | Rachel",
    "EXAVITQu4vr4xnSDxMaL | Bella",
    "AZnzlk1XvdvUeBnXmlld | Domi",
    "MF3mGyEYCl7XYWbV9V6O | Elli",
]


def parse_pool(text: str) -> List[str]:
    """Each line: 'voice_id' or 'voice_id | name'."""
    ids = []
    for line in text.splitlines():
        line = line.split("#")[0].strip()
        if line:
            ids.append(line.split("|")[0].strip())
    return ids


def list_account_voices(api_key: str) -> Tuple[List[str], List[str]]:
    from elevenlabs.client import ElevenLabs

    client = ElevenLabs(api_key=api_key)
    male, female = [], []
    for v in client.voices.get_all().voices:
        g = ((v.labels or {}).get("gender") or "").lower()
        line = f"{v.voice_id} | {v.name}"
        if g == "male":
            male.append(line)
        elif g == "female":
            female.append(line)
    return male, female


def assign_voices(speakers: List[dict], male_pool: List[str],
                  female_pool: List[str]) -> Tuple[Dict[str, str], List[str]]:
    """speaker -> voice_id. Same speaker always gets the same voice; different speakers
    get different voices until a pool runs out."""
    m = f = 0
    out: Dict[str, str] = {}
    warnings: List[str] = []
    for sp in speakers:
        override = (sp.get("voice_override") or "").strip()
        if override:
            out[sp["speaker"]] = override
            continue
        female = sp["gender"] == "Female"
        pool = female_pool if female else male_pool
        if not pool:
            raise ValueError(f"No {'female' if female else 'male'} voice IDs configured.")
        idx = f if female else m
        if idx >= len(pool):
            warnings.append(f"More {'female' if female else 'male'} speakers than voices - "
                            f"{sp['speaker']} reuses a voice. Add more voice IDs.")
        out[sp["speaker"]] = pool[idx % len(pool)]
        if female:
            f += 1
        else:
            m += 1
    return out, warnings


# --------------------------------------------------------------------------- #
# 7. Dubbed audio track (TTS + timing)
# --------------------------------------------------------------------------- #
SR = 24000


def _seed_for(speaker: str) -> int:
    return int(hashlib.md5(speaker.encode()).hexdigest()[:8], 16)


def tts_to_file(client, text: str, voice_id: str, model_id: str, seed: int, out_path: Path) -> None:
    from elevenlabs import VoiceSettings

    last = None
    for attempt in range(5):
        try:
            stream = client.text_to_speech.convert(
                voice_id=voice_id, text=text, model_id=model_id,
                output_format="mp3_44100_128", seed=seed,
                # fixed settings -> same voice character for every line of a speaker
                voice_settings=VoiceSettings(stability=0.5, similarity_boost=0.8,
                                             style=0.0, use_speaker_boost=True),
            )
            out_path.write_bytes(b"".join(stream))
            return
        except Exception as e:  # rate limit / transient
            last = e
            time.sleep(2 ** attempt * 1.5)
    raise RuntimeError(f"ElevenLabs TTS failed: {last}")


def _load_clip(path: Path):
    from pydub import AudioSegment
    from pydub.silence import detect_leading_silence

    clip = AudioSegment.from_file(path)
    lead = detect_leading_silence(clip, silence_threshold=-45)
    trail = detect_leading_silence(clip.reverse(), silence_threshold=-45)
    if len(clip) - lead - trail > 100:
        clip = clip[lead: len(clip) - trail]
    clip = clip.set_channels(1).set_frame_rate(SR)
    if clip.dBFS != float("-inf"):
        clip = clip.apply_gain(-20.0 - clip.dBFS)  # same loudness for every line
    return clip


def _speed_up(clip, factor: float, tmp: Path):
    from pydub import AudioSegment

    src, dst = tmp / "_in.wav", tmp / "_out.wav"
    clip.export(src, format="wav")
    run(["ffmpeg", "-y", "-i", src, "-filter:a", f"atempo={factor:.4f}", dst])
    return AudioSegment.from_file(dst).set_channels(1).set_frame_rate(SR)


def build_dub_track(api_key: str, segs: List[dict], voice_map: Dict[str, str], model_id: str,
                    total_seconds: float, workdir: Path, max_speedup: float = 1.6,
                    progress: Progress = _noop) -> Tuple[str, List[Tuple[float, float]]]:
    """Returns (dub_wav_path, placements[(start,end)])."""
    from elevenlabs.client import ElevenLabs
    from pydub import AudioSegment

    client = ElevenLabs(api_key=api_key, timeout=300)
    tts_dir = workdir / "tts"
    tts_dir.mkdir(exist_ok=True)
    track = np.zeros(int((total_seconds + 5) * SR), dtype=np.int32)
    placements: List[Tuple[float, float]] = []
    segs = sorted(segs, key=lambda s: s["start"])

    for i, s in enumerate(segs):
        text = (s.get("translated") or "").strip()
        progress(i / max(len(segs), 1), f"Voicing {i + 1}/{len(segs)} ({s['speaker']})")
        if not text:
            continue
        voice = voice_map[s["speaker"]]
        key = hashlib.md5(f"{text}|{voice}|{model_id}".encode()).hexdigest()[:12]
        f = tts_dir / f"{i:05d}_{key}.mp3"
        if not f.exists():
            tts_to_file(client, text, voice, model_id, _seed_for(s["speaker"]), f)
        clip = _load_clip(f)

        # --- fit into the time the original speaker used ---
        slot = max(s["end"] - s["start"], 0.3)
        next_start = segs[i + 1]["start"] if i + 1 < len(segs) else s["start"] + slot * 2
        room = max(next_start - s["start"] - 0.08, slot)      # may borrow the pause after it
        budget = min(room, slot * 1.35)                          # allow a little overrun
        dur = len(clip) / 1000.0
        if dur > budget:
            factor = min(dur / budget, max_speedup)
            clip = _speed_up(clip, factor, tts_dir)

        samples = np.array(clip.get_array_of_samples(), dtype=np.int32)
        a = int(s["start"] * SR)
        b = min(a + len(samples), len(track))
        track[a:b] += samples[: b - a]
        placements.append((s["start"], s["start"] + len(samples) / SR))

    track = np.clip(track, -32768, 32767).astype(np.int16)
    out = workdir / "dub_audio.wav"
    AudioSegment(data=track.tobytes(), sample_width=2, frame_rate=SR, channels=1).export(out, format="wav")
    return str(out), placements


# --------------------------------------------------------------------------- #
# 8. Optional background (music / effects) separation with Demucs
# --------------------------------------------------------------------------- #
def separate_background(video: str, workdir: Path) -> Optional[str]:
    wav = workdir / "orig_44k.wav"
    run(["ffmpeg", "-y", "-i", video, "-vn", "-ac", "2", "-ar", "44100", wav])
    run([sys.executable, "-m", "demucs", "--two-stems=vocals", "-o", workdir / "demucs", wav])
    found = list((workdir / "demucs").rglob("no_vocals.wav"))
    return str(found[0]) if found else None


def mix_background(dub_wav: str, bg_wav: str, volume: float, workdir: Path) -> str:
    out = str(workdir / "final_audio.wav")
    run(["ffmpeg", "-y", "-i", dub_wav, "-i", bg_wav, "-filter_complex",
         f"[1:a]volume={volume}[b];[0:a]aresample=44100[a];[a][b]amix=inputs=2:duration=first:normalize=0[o]",
         "-map", "[o]", out])
    return out


# --------------------------------------------------------------------------- #
# 9. Lip sync with Wav2Lip (chunked so any video length works)
# --------------------------------------------------------------------------- #
def plan_chunks(duration: float, placements: List[Tuple[float, float]], target: float) -> List[float]:
    """Cut points that fall in silence so no dubbed sentence is split between chunks."""
    free = []
    pl = sorted(placements)
    prev_end = 0.0
    for a, b in pl:
        if a > prev_end + 0.2:
            free.append((prev_end + a) / 2)
        prev_end = max(prev_end, b)
    cuts = [0.0]
    cursor = 0.0
    while duration - cursor > target * 1.5:
        cands = [t for t in free if cursor + target * 0.5 <= t <= cursor + target * 1.5]
        t = min(cands, key=lambda x: abs(x - (cursor + target))) if cands else cursor + target
        cuts.append(t)
        cursor = t
    cuts.append(duration)
    return cuts


def lipsync(video: str, dub_wav: str, placements: List[Tuple[float, float]], workdir: Path,
            wav2lip_dir: str, checkpoint: str, python_exe: str, chunk_sec: float = 60.0,
            resize_factor: int = 1, progress: Progress = _noop) -> Tuple[str, int]:
    """Returns (silent_lipsynced_video, number_of_chunks_without_a_detectable_face)."""
    dur = media_duration(video)
    W, H, fps = probe_video(video)
    cuts = plan_chunks(dur, placements, chunk_sec)
    ldir = workdir / "lip"
    ldir.mkdir(exist_ok=True)
    wdir, ck = str(Path(wav2lip_dir).resolve()), str(Path(checkpoint).resolve())
    norm_files, skipped = [], 0

    for i, (a, b) in enumerate(zip(cuts[:-1], cuts[1:])):
        progress(i / max(len(cuts) - 1, 1), f"Lip-syncing chunk {i + 1}/{len(cuts) - 1}")
        norm = ldir / f"norm_{i:04d}.mp4"
        if not norm.exists():
            v, w, o = ldir / f"v_{i:04d}.mp4", ldir / f"a_{i:04d}.wav", ldir / f"o_{i:04d}.mp4"
            run(["ffmpeg", "-y", "-ss", f"{a:.3f}", "-t", f"{b - a:.3f}", "-i", video, "-an",
                 "-c:v", "libx264", "-preset", "veryfast", "-crf", "17", "-pix_fmt", "yuv420p", v])
            run(["ffmpeg", "-y", "-ss", f"{a:.3f}", "-t", f"{b - a:.3f}", "-i", dub_wav,
                 "-ac", "1", "-ar", "16000", w])
            src = o
            try:
                run([python_exe, "inference.py", "--checkpoint_path", ck, "--face", v.resolve(),
                     "--audio", w.resolve(), "--outfile", o.resolve(),
                     "--resize_factor", resize_factor], cwd=wdir)
            except RuntimeError:        # no face found etc. -> keep the original picture
                src = v
                skipped += 1
            run(["ffmpeg", "-y", "-i", src, "-an", "-vf", f"scale={W}:{H},fps={fps:.3f}",
                 "-c:v", "libx264", "-preset", "veryfast", "-crf", "17", "-pix_fmt", "yuv420p", norm])
        norm_files.append(norm)

    lst = ldir / "list.txt"
    lst.write_text("".join(f"file '{p.resolve()}'\n" for p in norm_files))
    out = str(workdir / "lipsynced_silent.mp4")
    run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", out])
    return out, skipped


# --------------------------------------------------------------------------- #
# 10. Final mux
# --------------------------------------------------------------------------- #
def mux_video(video: str, audio: str, out_path: str) -> str:
    try:
        run(["ffmpeg", "-y", "-i", video, "-i", audio, "-map", "0:v:0", "-map", "1:a:0",
             "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", out_path])
    except RuntimeError:  # codec not allowed in mp4 -> re-encode
        run(["ffmpeg", "-y", "-i", video, "-i", audio, "-map", "0:v:0", "-map", "1:a:0",
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-c:a", "aac",
             "-b:a", "192k", "-shortest", out_path])
    return out_path
