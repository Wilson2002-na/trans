"""
Core AI dubbing pipeline (v2).

Only needs: ffmpeg, numpy, requests (+ deep-translator for translation).
ElevenLabs is called through its REST API directly, so SDK versions cannot break it.

Steps
-----
1. extract_audio        ffmpeg: video -> small mono mp3
2. transcribe           ElevenLabs Scribe: language detection + speakers + word timestamps
3. build_segments       words -> timed single-speaker sentences (when each person starts / stops)
4. detect_genders       pitch analysis per speaker -> Male / Female
5. translate_segments   Google Translate -> target language
6. assign_voices        one unique voice per speaker, fixed for the whole video
7. build_dub_track      ElevenLabs TTS per sentence, fitted to the original timing
8. separate_background  (optional) Demucs
9. lipsync              (optional) Wav2Lip in chunks cut at pauses
10. mux_video           ffmpeg: video + dubbed audio
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import time
import wave
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import requests

Progress = Callable[[float, str], None]
API = "https://api.elevenlabs.io/v1"


def _noop(p: float, m: str) -> None:
    pass


# --------------------------------------------------------------------------- #
# Languages
# --------------------------------------------------------------------------- #
# name -> (ISO 639-1, Google Translate code)
LANGUAGES: Dict[str, Tuple[str, str]] = {
    "English": ("en", "en"), "Tamil": ("ta", "ta"), "Hindi": ("hi", "hi"),
    "Malayalam": ("ml", "ml"), "Telugu": ("te", "te"), "Kannada": ("kn", "kn"),
    "Bengali": ("bn", "bn"), "Marathi": ("mr", "mr"), "Gujarati": ("gu", "gu"),
    "Punjabi": ("pa", "pa"), "Urdu": ("ur", "ur"), "Nepali": ("ne", "ne"),
    "Arabic": ("ar", "ar"), "Spanish": ("es", "es"), "French": ("fr", "fr"),
    "German": ("de", "de"), "Italian": ("it", "it"), "Portuguese": ("pt", "pt"),
    "Russian": ("ru", "ru"), "Japanese": ("ja", "ja"), "Korean": ("ko", "ko"),
    "Chinese": ("zh", "zh-CN"), "Indonesian": ("id", "id"), "Turkish": ("tr", "tr"),
    "Thai": ("th", "th"), "Vietnamese": ("vi", "vi"),
}
ISO1_TO_NAME = {v[0]: k for k, v in LANGUAGES.items()}

ISO3_TO_1 = {
    "eng": "en", "tam": "ta", "hin": "hi", "mal": "ml", "tel": "te", "kan": "kn",
    "ben": "bn", "mar": "mr", "guj": "gu", "pan": "pa", "urd": "ur", "nep": "ne",
    "ara": "ar", "spa": "es", "fra": "fr", "fre": "fr", "deu": "de", "ger": "de",
    "ita": "it", "por": "pt", "rus": "ru", "jpn": "ja", "kor": "ko", "zho": "zh",
    "chi": "zh", "ind": "id", "tur": "tr", "tha": "th", "vie": "vi",
}

# Languages covered by eleven_multilingual_v2; the rest of ours need eleven_v3.
V2_LANGS = {"en", "hi", "ta", "ar", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh", "id", "tr"}


def to_iso1(code: Optional[str]) -> str:
    if not code:
        return "unknown"
    code = code.lower().split("-")[0]
    return ISO3_TO_1.get(code, code)


def language_label(iso1: str) -> str:
    return f"{ISO1_TO_NAME.get(iso1, iso1)} ({iso1})"


def pick_tts_model(lang_code: str, choice: str = "Auto") -> str:
    if choice != "Auto":
        return choice
    return "eleven_multilingual_v2" if lang_code in V2_LANGS else "eleven_v3"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def run(cmd: list, cwd: Optional[str] = None) -> str:
    try:
        p = subprocess.run([str(c) for c in cmd], cwd=cwd, capture_output=True, text=True)
    except FileNotFoundError:
        raise RuntimeError(f"'{cmd[0]}' was not found. Install ffmpeg and make sure it is on PATH.")
    if p.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(map(str, cmd[:6]))} ...\n{p.stderr[-2000:]}")
    return p.stdout


def media_duration(path: str) -> float:
    out = run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
               "-of", "default=nw=1:nk=1", path])
    return float(out.strip())


def has_audio_stream(path: str) -> bool:
    out = run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
               "stream=index", "-of", "csv=p=0", path])
    return bool(out.strip())


def probe_video(path: str) -> Tuple[int, int, float]:
    out = run(["ffprobe", "-v", "error", "-select_streams", "v:0",
               "-show_entries", "stream=width,height,r_frame_rate",
               "-of", "csv=p=0", path]).strip()
    w, h, rate = out.split(",")[:3]
    num, den = rate.split("/") if "/" in rate else (rate, "1")
    return int(w), int(h), float(num) / float(den or 1)


def decode_pcm(path: str, sr: int, extra_filter: Optional[str] = None) -> np.ndarray:
    """Decode any audio/video file to mono float32 samples at `sr`, using ffmpeg."""
    cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", str(sr)]
    if extra_filter:
        cmd += ["-af", extra_filter]
    cmd += ["-f", "f32le", "-"]
    p = subprocess.run(cmd, capture_output=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg could not decode {path}:\n{p.stderr.decode(errors='ignore')[-1500:]}")
    return np.frombuffer(p.stdout, dtype=np.float32).copy()


def write_wav(path: Path, samples: np.ndarray, sr: int) -> None:
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def fmt_srt_time(t: float) -> str:
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def segments_to_srt(segs: List[dict], key: str = "translated") -> str:
    lines = []
    for i, s in enumerate(segs, 1):
        lines += [str(i), f"{fmt_srt_time(s['start'])} --> {fmt_srt_time(s['end'])}", str(s.get(key, "")), ""]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# ElevenLabs REST helpers
# --------------------------------------------------------------------------- #
class ElevenError(RuntimeError):
    pass


def _err_text(r: requests.Response) -> str:
    try:
        j = r.json()
        d = j.get("detail", j) if isinstance(j, dict) else j
        if isinstance(d, dict):
            d = d.get("message") or d.get("status") or json.dumps(d)
        return str(d)
    except Exception:
        return (r.text or "")[:300]


def _request(method: str, path: str, api_key: str, retries: int = 4, **kw) -> requests.Response:
    if not api_key:
        raise ElevenError("No ElevenLabs API key. Paste it in the sidebar and press Enter.")
    kw.setdefault("timeout", 60)
    last: Optional[Exception] = None
    for attempt in range(retries):
        try:
            r = requests.request(method, API + path, headers={"xi-api-key": api_key}, **kw)
        except requests.RequestException as e:
            last = ElevenError(f"Network error talking to ElevenLabs: {e}")
            time.sleep(2 ** attempt)
            continue
        if r.status_code in (429, 500, 502, 503, 504) and attempt < retries - 1:
            last = ElevenError(f"ElevenLabs error {r.status_code}: {_err_text(r)}")
            time.sleep(2 ** attempt * 1.5)
            continue
        if not r.ok:
            hint = ""
            if r.status_code == 401:
                hint = " (API key is wrong or missing a permission - create a key with Speech to Text, Text to Speech and Voices access)"
            elif r.status_code == 402:
                hint = " (your plan does not allow this - e.g. library voices need a paid plan)"
            raise ElevenError(f"ElevenLabs error {r.status_code}: {_err_text(r)}{hint}")
        return r
    raise last or ElevenError("ElevenLabs request failed")


def check_api_key(api_key: str) -> str:
    r = _request("GET", "/user/subscription", api_key, retries=1, timeout=30)
    j = r.json()
    return (f"Key works. Plan: {j.get('tier', '?')} - "
            f"{j.get('character_count', 0)}/{j.get('character_limit', 0)} characters used")


def list_account_voices(api_key: str) -> Tuple[List[str], List[str]]:
    r = _request("GET", "/voices", api_key, timeout=60)
    male, female = [], []
    for v in r.json().get("voices", []):
        g = ((v.get("labels") or {}).get("gender") or "").lower()
        line = f"{v['voice_id']} | {v.get('name', '')}"
        if g == "male":
            male.append(line)
        elif g == "female":
            female.append(line)
    return male, female


def voice_ok(api_key: str, voice_id: str) -> bool:
    try:
        _request("GET", f"/voices/{voice_id}", api_key, retries=1, timeout=30)
        return True
    except ElevenError as e:
        if "error 401" in str(e) or "Network error" in str(e):
            return True          # cannot tell - do not block the run
        return False


# --------------------------------------------------------------------------- #
# 1. Audio extraction
# --------------------------------------------------------------------------- #
def extract_audio(video: str, workdir: Path) -> str:
    if not has_audio_stream(video):
        raise RuntimeError("This video has no audio track.")
    out = workdir / "source_audio.mp3"
    if not out.exists() or out.stat().st_size == 0:
        run(["ffmpeg", "-y", "-i", video, "-vn", "-ac", "1", "-ar", "16000", "-b:a", "48k", out])
    return str(out)


# --------------------------------------------------------------------------- #
# 2. Transcription + speaker separation (ElevenLabs Scribe)
# --------------------------------------------------------------------------- #
def transcribe(api_key: str, audio_path: str, num_speakers: Optional[int] = None,
               cache_path: Optional[Path] = None, model_id: str = "scribe_v1") -> dict:
    """Returns {"language": iso1, "words": [{text,start,end,type,speaker_id}]}"""
    if cache_path and cache_path.exists():
        return json.loads(cache_path.read_text())

    data = {"model_id": model_id, "diarize": "true", "tag_audio_events": "false",
            "timestamps_granularity": "word"}            # no language_code -> auto-detect
    if num_speakers:
        data["num_speakers"] = str(int(num_speakers))
    audio_bytes = Path(audio_path).read_bytes()
    r = _request("POST", "/speech-to-text", api_key, data=data,
                 files={"file": (Path(audio_path).name, audio_bytes, "audio/mpeg")}, timeout=3600)
    j = r.json()

    words = []
    for w in j.get("words") or []:
        words.append({"text": w.get("text", ""), "start": w.get("start"), "end": w.get("end"),
                      "type": w.get("type", "word"), "speaker_id": w.get("speaker_id") or "speaker_0"})
    res = {"language": to_iso1(j.get("language_code")), "words": words,
           "language_probability": j.get("language_probability")}
    if cache_path:
        cache_path.write_text(json.dumps(res))
    return res


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
# 4. Gender detection (pitch based, numpy only)
# --------------------------------------------------------------------------- #
def estimate_f0(clip: np.ndarray, sr: int, fmin: float = 70.0, fmax: float = 400.0) -> np.ndarray:
    """Autocorrelation pitch tracker. Returns F0 (Hz) of the voiced frames."""
    frame, hop = int(0.04 * sr), int(0.02 * sr)
    if len(clip) < frame * 3:
        return np.array([])
    lag_min, lag_max = int(sr / fmax), int(sr / fmin)
    nfft = 1 << (2 * frame - 1).bit_length()
    rms_all = np.sqrt(np.mean(clip ** 2)) + 1e-9
    out = []
    win = np.hanning(frame)
    for i in range(0, len(clip) - frame, hop):
        x = clip[i:i + frame]
        if np.sqrt(np.mean(x ** 2)) < 0.5 * rms_all:      # skip quiet / unvoiced frames
            continue
        x = (x - x.mean()) * win
        spec = np.fft.rfft(x, nfft)
        ac = np.fft.irfft(spec * np.conj(spec))[:lag_max + 1]
        if ac[0] <= 0:
            continue
        ac = ac / ac[0]
        seg = ac[lag_min:lag_max + 1]
        peak = seg.max()
        if peak < 0.45:
            continue
        # first strong peak (avoids octave-down errors)
        cand = np.where((seg[1:-1] >= seg[:-2]) & (seg[1:-1] >= seg[2:]) & (seg[1:-1] >= 0.9 * peak))[0]
        lag = lag_min + 1 + (cand[0] if len(cand) else int(np.argmax(seg)) - 1)
        out.append(sr / lag)
    return np.array(out)


def detect_genders(audio_path: str, segs: List[dict], threshold_hz: float = 165.0,
                   progress: Progress = _noop) -> List[dict]:
    """One row per speaker, in order of first appearance."""
    sr = 16000
    y = decode_pcm(audio_path, sr)
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
            f0 = estimate_f0(clip, sr)
            if len(f0):
                f0s.append(f0)
            used += len(clip) / sr
        arr = np.concatenate(f0s) if f0s else np.array([])
        med = float(np.median(arr)) if len(arr) > 10 else 0.0
        gender = "Female" if med >= threshold_hz else "Male"
        rows.append({"speaker": spk, "gender": gender, "median_f0_hz": round(med, 1),
                     "speaking_seconds": round(speaking, 1), "voice_override": ""})
    return rows


# --------------------------------------------------------------------------- #
# 5. Translation
# --------------------------------------------------------------------------- #
def _gtx_translate(text: str, tgt: str) -> str:
    """Fallback: Google's public web endpoint."""
    r = requests.get("https://translate.googleapis.com/translate_a/single",
                     params={"client": "gtx", "sl": "auto", "tl": tgt, "dt": "t", "q": text}, timeout=30)
    r.raise_for_status()
    return "".join(part[0] for part in r.json()[0] if part and part[0])


def translate_segments(segs: List[dict], target_name: str, source_iso: str,
                       progress: Progress = _noop) -> Tuple[List[dict], int]:
    """Returns (segments with 'translated', number of sentences that could not be translated)."""
    tgt_iso, tgt_g = LANGUAGES[target_name]
    out = [dict(s) for s in segs]
    if source_iso == tgt_iso:
        for s in out:
            s["translated"] = s["text"]
        return out, 0

    translator = None
    try:
        from deep_translator import GoogleTranslator
        translator = GoogleTranslator(source="auto", target=tgt_g)
    except Exception:
        translator = None

    failed, last_err = 0, None
    for i, s in enumerate(out):
        text = str(s["text"]).strip()
        res = None
        for attempt in range(3):
            try:
                if translator is not None:
                    res = translator.translate(text)
                if not res:
                    res = _gtx_translate(text, tgt_g)
                break
            except Exception as e:
                last_err = e
                try:
                    res = _gtx_translate(text, tgt_g)
                    break
                except Exception as e2:
                    last_err = e2
                    time.sleep(1.0 * (attempt + 1))
        if res:
            s["translated"] = res
        else:
            s["translated"] = s["text"]
            failed += 1
        progress((i + 1) / len(out), f"Translating {i + 1}/{len(out)}")
    if failed == len(out):
        raise RuntimeError(f"Translation failed for every sentence - check the internet connection. ({last_err})")
    return out, failed


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
    ids = []
    for line in text.splitlines():
        line = line.split("#")[0].strip()
        if line:
            ids.append(line.split("|")[0].strip())
    return ids


def assign_voices(speakers: List[dict], male_pool: List[str],
                  female_pool: List[str]) -> Tuple[Dict[str, str], List[str]]:
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
            raise ValueError(f"No {'female' if female else 'male'} voice IDs configured - add some in the sidebar.")
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


def fix_invalid_voices(api_key: str, voice_map: Dict[str, str],
                       speakers: List[dict]) -> Tuple[Dict[str, str], List[str]]:
    """Replace voice IDs that do not exist for this account with unused voices of the same gender."""
    notes: List[str] = []
    bad = [s for s, v in voice_map.items() if not voice_ok(api_key, v)]
    if not bad:
        return voice_map, notes
    male, female = list_account_voices(api_key)
    pools = {"Male": parse_pool("\n".join(male)), "Female": parse_pool("\n".join(female))}
    gender = {s["speaker"]: s["gender"] for s in speakers}
    used = set(voice_map.values())
    out = dict(voice_map)
    for spk in bad:
        cand = [v for v in pools.get(gender.get(spk, "Male"), []) if v not in used]
        if not cand:
            raise ValueError(f"Voice {voice_map[spk]} (for {spk}) is not available on your account and no "
                             f"replacement {gender.get(spk, '')} voice was found. Add voices to My Voices "
                             f"on elevenlabs.io and paste their IDs in the sidebar.")
        out[spk] = cand[0]
        used.add(cand[0])
        notes.append(f"Voice {voice_map[spk]} not available - {spk} now uses {cand[0]} from your account.")
    return out, notes


# --------------------------------------------------------------------------- #
# 7. Dubbed audio track (TTS + timing)  - numpy + ffmpeg only
# --------------------------------------------------------------------------- #
SR = 24000


def _seed_for(speaker: str) -> int:
    return int(hashlib.md5(speaker.encode()).hexdigest()[:8], 16)


def tts_to_file(api_key: str, text: str, voice_id: str, model_id: str, seed: int, out_path: Path) -> None:
    full = {"text": text, "model_id": model_id, "seed": seed,
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.8, "style": 0.0,
                               "use_speaker_boost": True}}
    minimal = {"text": text, "model_id": model_id}
    try:
        r = _request("POST", f"/text-to-speech/{voice_id}", api_key, json=full,
                     params={"output_format": "mp3_44100_128"}, timeout=300)
    except ElevenError as e:
        if "error 422" not in str(e):
            raise
        r = _request("POST", f"/text-to-speech/{voice_id}", api_key, json=minimal,
                     params={"output_format": "mp3_44100_128"}, timeout=300)
    if not r.content:
        raise ElevenError("ElevenLabs returned empty audio.")
    out_path.write_bytes(r.content)


def _prepare_clip(path: Path) -> np.ndarray:
    """Decode, trim leading/trailing silence, normalise loudness."""
    x = decode_pcm(str(path), SR)
    if len(x) == 0:
        return x
    thr = 10 ** (-45 / 20)
    idx = np.where(np.abs(x) > thr)[0]
    if len(idx) and idx[-1] - idx[0] > SR * 0.1:
        x = x[idx[0]: idx[-1] + 1]
    rms = float(np.sqrt(np.mean(x ** 2)))
    if rms > 1e-5:
        x = x * min(0.1 / rms, 8.0)          # about -20 dBFS RMS, same loudness for every line
    return np.clip(x, -1.0, 1.0)


def _tempo(x: np.ndarray, factor: float, tmp: Path) -> np.ndarray:
    """Speed up speech without changing pitch (ffmpeg atempo)."""
    src = tmp / "_tempo_in.wav"
    write_wav(src, x, SR)
    return decode_pcm(str(src), SR, extra_filter=f"atempo={factor:.4f}")


def build_dub_track(api_key: str, segs: List[dict], voice_map: Dict[str, str], model_id: str,
                    total_seconds: float, workdir: Path, max_speedup: float = 1.6,
                    progress: Progress = _noop) -> Tuple[str, List[Tuple[float, float]]]:
    """Returns (dub_wav_path, placements[(start,end)])."""
    tts_dir = workdir / "tts"
    tts_dir.mkdir(exist_ok=True)
    track = np.zeros(int((total_seconds + 5) * SR), dtype=np.float32)
    placements: List[Tuple[float, float]] = []
    segs = sorted(segs, key=lambda s: s["start"])
    spoken = 0

    for i, s in enumerate(segs):
        text = str(s.get("translated") or "").strip()
        progress(i / max(len(segs), 1), f"Voicing {i + 1}/{len(segs)} ({s['speaker']})")
        if not text:
            continue
        voice = voice_map[s["speaker"]]
        key = hashlib.md5(f"{text}|{voice}|{model_id}".encode()).hexdigest()[:12]
        f = tts_dir / f"{i:05d}_{key}.mp3"
        if not f.exists():
            tts_to_file(api_key, text, voice, model_id, _seed_for(s["speaker"]), f)
        clip = _prepare_clip(f)
        if len(clip) == 0:
            continue

        # fit into the time the original speaker used
        slot = max(s["end"] - s["start"], 0.3)
        next_start = segs[i + 1]["start"] if i + 1 < len(segs) else s["start"] + slot * 2
        room = max(next_start - s["start"] - 0.08, slot)      # may borrow the pause after it
        budget = min(room, slot * 1.35)
        dur = len(clip) / SR
        if dur > budget:
            clip = _tempo(clip, min(dur / budget, max_speedup), tts_dir)

        a = int(s["start"] * SR)
        b = min(a + len(clip), len(track))
        if b > a:
            track[a:b] += clip[: b - a]
        placements.append((s["start"], s["start"] + len(clip) / SR))
        spoken += 1

    if spoken == 0:
        raise RuntimeError("Nothing to dub - every translated line is empty.")
    out = workdir / "dub_audio.wav"
    write_wav(out, track, SR)
    return str(out), placements


# --------------------------------------------------------------------------- #
# 8. Optional background (music / effects) with Demucs
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
    free, prev_end = [], 0.0
    for a, b in sorted(placements):
        if a > prev_end + 0.2:
            free.append((prev_end + a) / 2)
        prev_end = max(prev_end, b)
    cuts, cursor = [0.0], 0.0
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
    """Returns (silent_lipsynced_video, chunks_without_a_detectable_face)."""
    if not (Path(wav2lip_dir) / "inference.py").exists():
        raise RuntimeError(f"Wav2Lip not found in '{wav2lip_dir}' (inference.py missing). See README section 3.")
    if not Path(checkpoint).exists():
        raise RuntimeError(f"Wav2Lip checkpoint not found: {checkpoint}")
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
            except RuntimeError:
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
    except RuntimeError:
        run(["ffmpeg", "-y", "-i", video, "-i", audio, "-map", "0:v:0", "-map", "1:a:0",
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-c:a", "aac",
             "-b:a", "192k", "-shortest", out_path])
    return out_path
