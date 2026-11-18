import base64
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import requests
import soundfile as sf
from faster_whisper import WhisperModel
from pydub import AudioSegment
from deep_translator import GoogleTranslator

LANGUAGES = {
    "Tamil": {"translator": "ta", "eleven": "ta"},
    "English": {"translator": "en", "eleven": "en"},
    "Hindi": {"translator": "hi", "eleven": "hi"},
    "Malayalam": {"translator": "ml", "eleven": "ml"},
    "Telugu": {"translator": "te", "eleven": "te"},
    "Kannada": {"translator": "kn", "eleven": "kn"},
    "Bengali": {"translator": "bn", "eleven": "bn"},
    "Marathi": {"translator": "mr", "eleven": "mr"},
    "Gujarati": {"translator": "gu", "eleven": "gu"},
    "Punjabi": {"translator": "pa", "eleven": "pa"},
    "Spanish": {"translator": "es", "eleven": "es"},
    "French": {"translator": "fr", "eleven": "fr"},
    "German": {"translator": "de", "eleven": "de"},
    "Japanese": {"translator": "ja", "eleven": "ja"},
    "Korean": {"translator": "ko", "eleven": "ko"},
    "Arabic": {"translator": "ar", "eleven": "ar"},
    "Portuguese": {"translator": "pt", "eleven": "pt"},
    "Russian": {"translator": "ru", "eleven": "ru"},
    "Chinese (Simplified)": {"translator": "zh-CN", "eleven": "zh"},
}

def run(cmd):
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"Command failed:\n{' '.join(map(str, cmd))}\n{result.stderr[-4000:]}")
    return result

def extract_audio(video_path, audio_path):
    run(["ffmpeg", "-y", "-i", str(video_path), "-vn", "-ac", "1", "-ar", "16000",
         "-c:a", "pcm_s16le", str(audio_path)])

def transcribe_audio(audio_path, model_size="small"):
    # CPU mode is broadly compatible; set device="cuda", compute_type="float16" for a supported GPU.
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments_iter, info = model.transcribe(str(audio_path), word_timestamps=True, vad_filter=True)
    output = []
    for seg in segments_iter:
        text = seg.text.strip()
        if text:
            output.append({"start": float(seg.start), "end": float(seg.end), "text": text})
    return output, info.language

def diarize_audio(audio_path, hf_token):
    try:
        from pyannote.audio import Pipeline
    except ImportError as e:
        raise RuntimeError("Install pyannote.audio using the optional diarization requirements.") from e
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1", token=hf_token)
    result = pipeline(str(audio_path))
    turns = []
    annotation = getattr(result, "speaker_diarization", result)
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        turns.append({"start": float(turn.start), "end": float(turn.end), "speaker": str(speaker)})
    return turns

def assign_speakers(transcript, turns):
    if not turns:
        return [{**s, "speaker": "SPEAKER_00", "translation": ""} for s in transcript]
    assigned = []
    for seg in transcript:
        best_speaker, best_overlap = "SPEAKER_00", 0.0
        for turn in turns:
            overlap = max(0.0, min(seg["end"], turn["end"]) - max(seg["start"], turn["start"]))
            if overlap > best_overlap:
                best_overlap, best_speaker = overlap, turn["speaker"]
        assigned.append({**seg, "speaker": best_speaker, "translation": ""})
    return assigned

def translate_segments(segments, target_language):
    target = LANGUAGES[target_language]["translator"]
    for seg in segments:
        if not seg.get("translation"):
            seg["translation"] = GoogleTranslator(source="auto", target=target).translate(seg["text"])
    return segments

def get_elevenlabs_voices(api_key):
    response = requests.get("https://api.elevenlabs.io/v1/voices",
                            headers={"xi-api-key": api_key}, timeout=60)
    response.raise_for_status()
    return [{"name": v.get("name", "Voice"),
             "voice_id": v["voice_id"],
             "labels": v.get("labels", {})}
            for v in response.json().get("voices", [])]

def elevenlabs_tts(text, voice_id, api_key, language_code):
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    response = requests.post(
        url,
        headers={"xi-api-key": api_key, "Content-Type": "application/json"},
        params={"output_format": "mp3_44100_128"},
        json={"text": text, "model_id": "eleven_multilingual_v2",
              "language_code": language_code,
              "voice_settings": {"stability": 0.55, "similarity_boost": 0.8}},
        timeout=180,
    )
    response.raise_for_status()
    return response.content

def fit_audio_to_duration(audio, duration_ms):
    if len(audio) <= duration_ms:
        return audio
    # Speed up moderately to fit the original speech interval. Avoid extreme acceleration.
    ratio = len(audio) / max(duration_ms, 1)
    if ratio > 1.35:
        raise RuntimeError(
            "A translated line is over 35% longer than its original time slot. "
            "Shorten the translation or allow a longer timing window."
        )
    return audio.speedup(playback_speed=ratio, chunk_size=100, crossfade=20)

def synthesize_segments(segments, voice_map, api_key, output_path):
    if not segments:
        raise RuntimeError("No speech segments were detected.")
    end_ms = int(max(s["end"] for s in segments) * 1000) + 1000
    mix = AudioSegment.silent(duration=end_ms, frame_rate=44100).set_channels(2)
    for index, seg in enumerate(segments):
        text = (seg.get("translation") or "").strip()
        if not text:
            continue
        voice_id = voice_map[seg["speaker"]]
        # ElevenLabs returns one audio clip for each timed utterance. It is placed at
        # the detected start time, retaining pauses between original utterances.
        mp3_bytes = elevenlabs_tts(text, voice_id, api_key, "en")
        temp_mp3 = Path(tempfile.mkstemp(suffix=".mp3")[1])
        try:
            temp_mp3.write_bytes(mp3_bytes)
            clip = AudioSegment.from_file(temp_mp3, format="mp3").set_frame_rate(44100).set_channels(2)
        finally:
            temp_mp3.unlink(missing_ok=True)
        slot_ms = max(1, int((seg["end"] - seg["start"]) * 1000))
        clip = fit_audio_to_duration(clip, slot_ms)
        mix = mix.overlay(clip, position=max(0, int(seg["start"] * 1000)))
    mix.export(output_path, format="wav")

def assemble_dubbed_video(video_path, dubbed_audio, output_path, keep_original=False, original_volume=0.1):
    if keep_original and original_volume > 0:
        with tempfile.TemporaryDirectory() as tmp:
            original = Path(tmp) / "original.wav"
            extract_audio(video_path, original)
            bg = AudioSegment.from_file(original).set_frame_rate(44100).set_channels(2)
            dub = AudioSegment.from_file(dubbed_audio).set_frame_rate(44100).set_channels(2)
            bg = bg - (20 * (1 - original_volume))
            combined = bg.overlay(dub)
            mixed = Path(tmp) / "mixed.wav"
            combined.export(mixed, format="wav")
            run(["ffmpeg", "-y", "-i", str(video_path), "-i", str(mixed),
                 "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
                 "-c:a", "aac", "-shortest", str(output_path)])
    else:
        run(["ffmpeg", "-y", "-i", str(video_path), "-i", str(dubbed_audio),
             "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
             "-c:a", "aac", "-shortest", str(output_path)])

def run_wav2lip(repo_dir, checkpoint, video_path, output_path):
    repo = Path(repo_dir).expanduser().resolve()
    checkpoint = Path(checkpoint).expanduser().resolve()
    inference = repo / "inference.py"
    if not inference.exists():
        raise FileNotFoundError(f"Wav2Lip inference.py not found in {repo}. Clone the official repository first.")
    if not checkpoint.exists():
        raise FileNotFoundError(f"Wav2Lip checkpoint not found: {checkpoint}")
    raise RuntimeError(
        "Wav2Lip needs the generated dubbed WAV as its --audio input. "
        "This UI helper currently requires wiring that WAV path into the call. "
        "Use the command shown in README after generating dubbed.wav."
    )
