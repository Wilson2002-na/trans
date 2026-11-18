import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from dubbing import (
    LANGUAGES, extract_audio, transcribe_audio, diarize_audio,
    assign_speakers, translate_segments, get_elevenlabs_voices,
    synthesize_segments, assemble_dubbed_video, run_wav2lip,
)

load_dotenv()
st.set_page_config(page_title="Multilingual Video Dubber", page_icon="🎬", layout="wide")
st.title("🎬 Multilingual AI Video Dubber")
st.caption("Transcribe • identify speakers • translate • generate consistent voices • synchronize")

with st.sidebar:
    st.header("API credentials")
    eleven_key = st.text_input("sk_155bd03b99285b4dfecfb19008f1f347bc5e113f446329e7", value=os.getenv("ELEVENLABS_API_KEY", ""), type="password")
    hf_token = st.text_input("Hugging Face token (speaker diarization)", value=os.getenv("HF_TOKEN", ""), type="password")
    st.divider()
    st.header("Processing")
    model_size = st.selectbox("Whisper model", ["small", "medium", "large-v3"], index=0)
    use_diarization = st.checkbox("Identify multiple speakers (recommended)", value=True)
    keep_original_bg = st.checkbox("Keep original audio quietly under dubbing", value=False)
    bg_volume = st.slider("Original audio volume", 0.0, 0.5, 0.12, 0.01) if keep_original_bg else 0.0
    st.caption("Long videos need substantial CPU/GPU time and API credits.")

uploaded = st.file_uploader("Upload a video", type=["mp4", "mov", "mkv", "avi", "webm", "m4v"])
c1, c2 = st.columns(2)
with c1:
    target_language = st.selectbox("Dub into", list(LANGUAGES.keys()), index=0)
with c2:
    st.info("Source language is detected automatically by Whisper. You can review the transcript before generating audio.")

if uploaded:
    st.video(uploaded)
    if st.button("1. Analyse video", type="primary", use_container_width=True):
        if not eleven_key:
            st.error("Add your ElevenLabs API key in the sidebar.")
            st.stop()
        work = Path(tempfile.mkdtemp(prefix="video_dubber_"))
        video_path = work / uploaded.name
        video_path.write_bytes(uploaded.getbuffer())
        audio_path = work / "source.wav"
        with st.status("Analysing video...", expanded=True) as status:
            st.write("Extracting audio with FFmpeg...")
            extract_audio(video_path, audio_path)
            st.write("Transcribing and detecting source language...")
            transcript, detected_language = transcribe_audio(audio_path, model_size)
            st.write(f"Detected language: **{detected_language}**")
            if use_diarization:
                if not hf_token:
                    st.warning("No Hugging Face token supplied. Using one-speaker fallback.")
                    turns = []
                else:
                    st.write("Detecting speaker turns...")
                    turns = diarize_audio(audio_path, hf_token)
            else:
                turns = []
            segments = assign_speakers(transcript, turns)
            st.session_state["job"] = {
                "work": str(work), "video": str(video_path), "audio": str(audio_path),
                "segments": segments, "detected": detected_language,
                "target": target_language, "keep_bg": keep_original_bg, "bg_volume": bg_volume,
            }
            status.update(label="Analysis complete", state="complete")
        st.success(f"Found {len(segments)} speech segments.")

job = st.session_state.get("job")
if job:
    st.subheader("2. Review transcript and speakers")
    st.write(f"Source: **{job['detected']}** → Target: **{job['target']}**")
    edited = []
    for i, seg in enumerate(job["segments"]):
        with st.expander(f"{i+1}. {seg['start']:.2f}s–{seg['end']:.2f}s · {seg['speaker']}"):
            st.write(seg["text"])
            translated = st.text_area("Translation (editable)", value=seg.get("translation", ""), key=f"tr_{i}")
            edited.append({**seg, "translation": translated})
    st.session_state["edited_segments"] = edited

    if st.button("2. Translate all segments", use_container_width=True):
        from deep_translator import GoogleTranslator
        with st.spinner("Translating each speech segment..."):
            for seg in edited:
                seg["translation"] = GoogleTranslator(
                    source="auto", target=LANGUAGES[job["target"]]["translator"]
                ).translate(seg["text"])
        st.session_state["edited_segments"] = edited
        st.rerun()

    st.subheader("3. Assign a consistent ElevenLabs voice to each speaker")
    if st.button("Load my ElevenLabs voices"):
        try:
            st.session_state["voices"] = get_elevenlabs_voices(eleven_key)
        except Exception as e:
            st.error(f"Could not load voices: {e}")
    voices = st.session_state.get("voices", [])
    speakers = sorted({s["speaker"] for s in job["segments"]})
    voice_map = {}
    if voices:
        options = {f"{v['name']} · {v['voice_id']}": v["voice_id"] for v in voices}
        for speaker in speakers:
            voice_map[speaker] = st.selectbox(f"{speaker} voice", list(options.keys()), key=f"voice_{speaker}")
            voice_map[speaker] = options[voice_map[speaker]]
    else:
        st.warning("Load your ElevenLabs voices to assign different voices. Each speaker keeps the same selected voice throughout the video.")

    st.subheader("4. Generate dubbed video")
    st.caption("ElevenLabs voice selection is explicit so you can choose a suitable masculine/feminine-sounding voice. Pitch alone cannot reliably determine a person's gender.")
    if st.button("Generate dubbed video", type="primary", use_container_width=True):
        if not voices or not voice_map:
            st.error("Load voices and assign one voice to every speaker first.")
            st.stop()
        segments = st.session_state.get("edited_segments", edited)
        with st.status("Generating dubbed video...", expanded=True) as status:
            status.write("Translating any remaining empty segments...")
            segments = translate_segments(segments, job["target"])
            status.write("Generating speech for each speaker...")
            dub_audio = Path(job["work"]) / "dubbed.wav"
            synthesize_segments(segments, voice_map, eleven_key, dub_audio)
            status.write("Combining speech while preserving original timestamps and pauses...")
            output = Path(job["work"]) / "dubbed_video.mp4"
            assemble_dubbed_video(job["video"], dub_audio, output,
                                  job["keep_bg"], job["bg_volume"])
            status.update(label="Dubbed video ready", state="complete")
        st.session_state["output"] = str(output)

    if st.session_state.get("output"):
        output = Path(st.session_state["output"])
        st.subheader("5. Preview and download")
        st.video(str(output))
        st.download_button("Download dubbed MP4", data=output.read_bytes(),
                           file_name="dubbed_video.mp4", mime="video/mp4", use_container_width=True)

st.divider()
st.subheader("Optional: Wav2Lip lip synchronization")
st.warning("Wav2Lip is a separate, heavyweight step. It works best with a clearly visible, mostly frontal face. It does not guarantee accurate multi-person or off-screen-speaker lip sync.")
with st.expander("Run Wav2Lip on the generated video"):
    wav_repo = st.text_input("Local Wav2Lip repository path", value=os.getenv("WAV2LIP_DIR", "Wav2Lip"))
    checkpoint = st.text_input("Wav2Lip checkpoint path", value=os.getenv("WAV2LIP_CHECKPOINT", "checkpoints/wav2lip_gan.pth"))
    if st.button("Apply Wav2Lip"):
        if not st.session_state.get("output"):
            st.error("Generate a dubbed video first.")
        else:
            with st.spinner("Running Wav2Lip. This may take a long time..."):
                try:
                    lip_output = Path(st.session_state["output"]).with_name("dubbed_lipsynced.mp4")
                    run_wav2lip(wav_repo, checkpoint, st.session_state["output"], lip_output)
                    st.video(str(lip_output))
                    st.download_button("Download lip-synced MP4", lip_output.read_bytes(),
                                       file_name="dubbed_lipsynced.mp4", mime="video/mp4")
                except Exception as e:
                    st.error(str(e))
