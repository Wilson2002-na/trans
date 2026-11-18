"""AI Video Dubbing Studio - Streamlit UI.   Run:  streamlit run app.py"""
import hashlib
import os
from pathlib import Path

import pandas as pd
import streamlit as st

import pipeline as P

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

st.set_page_config(page_title="AI Video Dubbing Studio", page_icon="🎙️", layout="wide")
WORK = Path(os.getenv("DUB_WORKDIR", "work"))
WORK.mkdir(exist_ok=True)

# ---------------------------------------------------------------- session defaults
ss = st.session_state
ss.setdefault("male_pool", "\n".join(P.DEFAULT_MALE_VOICES))
ss.setdefault("female_pool", "\n".join(P.DEFAULT_FEMALE_VOICES))


def _load_account_voices():
    try:
        m, f = P.list_account_voices(ss["api_key"])
        if m:
            ss["male_pool"] = "\n".join(m)
        if f:
            ss["female_pool"] = "\n".join(f)
        ss["voice_msg"] = f"Loaded {len(m)} male and {len(f)} female voices."
    except Exception as e:
        ss["voice_msg"] = f"Could not load voices: {e}"


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    st.text_input("ElevenLabs API key", type="password", key="api_key",
                  value=os.getenv("ELEVENLABS_API_KEY", ""))
    target = st.selectbox("Dub into", list(P.LANGUAGES), index=1)
    tts_choice = st.selectbox("Voice model", ["Auto", "eleven_multilingual_v2", "eleven_v3",
                                              "eleven_turbo_v2_5"],
                              help="Auto: multilingual_v2 where supported, eleven_v3 for "
                                   "Malayalam, Telugu, Kannada, Bengali, Marathi, etc.")
    n_spk = st.number_input("Number of speakers (0 = auto detect)", 0, 32, 0)

    st.subheader("🗣️ Voice pools")
    st.caption("One voice ID per line (`id | name`). Each speaker gets their own voice from "
               "the pool matching their gender, fixed for the whole video.")
    st.button("Load voices from my ElevenLabs account", on_click=_load_account_voices,
              disabled=not ss.get("api_key"))
    if ss.get("voice_msg"):
        st.caption(ss["voice_msg"])
    st.text_area("Male voices", key="male_pool", height=120)
    st.text_area("Female voices", key="female_pool", height=120)

    st.subheader("🎬 Output")
    max_speed = st.slider("Max speed-up to fit timing", 1.0, 2.0, 1.5, 0.05)
    keep_bg = st.checkbox("Keep background music/effects (needs Demucs)")
    bg_vol = st.slider("Background volume", 0.1, 1.5, 0.8, 0.05, disabled=not keep_bg)

    do_lip = st.checkbox("Lip-sync with Wav2Lip (needs GPU + setup)")
    w2l_dir = st.text_input("Wav2Lip folder", os.getenv("WAV2LIP_DIR", "Wav2Lip"), disabled=not do_lip)
    w2l_ckpt = st.text_input("Checkpoint", os.getenv("WAV2LIP_CHECKPOINT",
                             "Wav2Lip/checkpoints/wav2lip_gan.pth"), disabled=not do_lip)
    w2l_py = st.text_input("Python for Wav2Lip", os.getenv("WAV2LIP_PYTHON", "python"),
                           disabled=not do_lip,
                           help="Path to the python of the Wav2Lip virtualenv (see README).")
    chunk = st.slider("Lip-sync chunk length (s)", 20, 180, 60, disabled=not do_lip)
    resize = st.selectbox("Wav2Lip resize factor", [1, 2, 4], disabled=not do_lip,
                          help="1 = full quality (slower, more VRAM). Higher = faster, lower quality.")

# ---------------------------------------------------------------- header + upload
st.title("🎙️ AI Video Dubbing Studio")
st.caption("Any language → Tamil · English · Hindi · Malayalam · Telugu · Kannada and more. "
           "Detects speakers & gender, keeps each speaker's voice consistent, matches timing, "
           "optionally lip-syncs with Wav2Lip.")

up = st.file_uploader("Upload a video", type=["mp4", "mov", "mkv", "avi", "webm", "m4v"])
path_in = st.text_input("…or path to a (long) video file on this machine")

video_path = None
if up is not None:
    pid = hashlib.md5(f"{up.name}{up.size}".encode()).hexdigest()[:10]
    pdir = WORK / pid
    pdir.mkdir(exist_ok=True)
    video_path = pdir / up.name
    if not video_path.exists():
        video_path.write_bytes(up.getbuffer())
elif path_in and Path(path_in).is_file():
    p = Path(path_in)
    pid = hashlib.md5(f"{p.name}{p.stat().st_size}".encode()).hexdigest()[:10]
    pdir = WORK / pid
    pdir.mkdir(exist_ok=True)
    video_path = p

if video_path:
    st.video(str(video_path))


def bar_cb(bar, msg, lo, hi):
    def cb(frac, text):
        bar.progress(min(max(lo + (hi - lo) * frac, 0.0), 1.0))
        msg.write(text)
    return cb


# ---------------------------------------------------------------- step 1: analyse
st.header("1 · Analyse")
if st.button("Detect language, speakers & gender, then translate", type="primary",
             disabled=not (video_path and ss.get("api_key"))):
    bar, msg = st.progress(0.0), st.empty()
    try:
        msg.write("Extracting audio…")
        audio = P.extract_audio(str(video_path), pdir)
        ss["duration"] = P.media_duration(str(video_path))
        bar.progress(0.05)

        msg.write("Transcribing, detecting language and separating speakers (ElevenLabs Scribe)…")
        tr = P.transcribe(ss["api_key"], audio, int(n_spk) or None)
        segs = P.build_segments(tr["words"])
        if not segs:
            st.error("No speech detected in this video.")
            st.stop()
        bar.progress(0.4)

        spk_rows = P.detect_genders(audio, segs, progress=bar_cb(bar, msg, 0.4, 0.65))
        segs = P.translate_segments(segs, target, tr["language"], bar_cb(bar, msg, 0.65, 1.0))
        ss.update(pdir=str(pdir), video=str(video_path), src_lang=tr["language"],
                  segs=pd.DataFrame(segs), speakers=pd.DataFrame(spk_rows), tgt=target)
        bar.progress(1.0)
        msg.success(f"Detected language: **{tr['language']}** · {len(spk_rows)} speaker(s) · "
                    f"{len(segs)} sentences")
    except Exception as e:
        st.error(f"Analysis failed: {e}")

# ---------------------------------------------------------------- step 2: review
if "segs" in ss:
    st.header("2 · Review (optional edits)")
    c1, c2 = st.columns([1, 2])
    with c1:
        st.subheader("Speakers")
        spk_df = st.data_editor(
            ss["speakers"], hide_index=True, use_container_width=True, key="spk_editor",
            disabled=["speaker", "median_f0_hz", "speaking_seconds"],
            column_config={
                "gender": st.column_config.SelectboxColumn("gender", options=["Male", "Female"]),
                "voice_override": st.column_config.TextColumn("voice ID override (optional)"),
            })
        st.caption("Gender comes from pitch analysis - fix it here if wrong. "
                   "Overriding a voice ID pins that speaker to a specific voice.")
    with c2:
        st.subheader(f"Script ({ss['src_lang']} → {ss['tgt']})")
        seg_df = st.data_editor(
            ss["segs"], hide_index=True, use_container_width=True, key="seg_editor", height=340,
            column_config={
                "start": st.column_config.NumberColumn("start (s)", format="%.2f"),
                "end": st.column_config.NumberColumn("end (s)", format="%.2f"),
                "text": st.column_config.TextColumn("original", disabled=True, width="medium"),
                "translated": st.column_config.TextColumn("dubbed text", width="large"),
            })

    if target != ss["tgt"] and st.button(f"Re-translate to {target}"):
        rows = P.translate_segments(seg_df.to_dict("records"), target, ss["src_lang"])
        ss["segs"], ss["tgt"] = pd.DataFrame(rows), target
        st.rerun()

    srt = P.segments_to_srt(seg_df.to_dict("records"))
    st.download_button("Download subtitles (.srt)", srt, "dubbed_subtitles.srt")

    # ------------------------------------------------------------ step 3: dub
    st.header("3 · Generate dubbed video")
    if st.button("Generate dub", type="primary", disabled=not ss.get("api_key")):
        bar, msg = st.progress(0.0), st.empty()
        try:
            pdir = Path(ss["pdir"])
            speakers = spk_df.to_dict("records")
            voice_map, warns = P.assign_voices(
                speakers, P.parse_pool(ss["male_pool"]), P.parse_pool(ss["female_pool"]))
            for w in warns:
                st.warning(w)
            st.write("Voice assignment:", voice_map)

            lang_iso = P.LANGUAGES[ss["tgt"]][0]
            model = P.pick_tts_model(lang_iso, tts_choice)
            msg.write(f"Voice model: `{model}`")
            segs = sorted(seg_df.to_dict("records"), key=lambda s: s["start"])

            wav, placements = P.build_dub_track(
                ss["api_key"], segs, voice_map, model, ss["duration"], pdir,
                max_speedup=max_speed, progress=bar_cb(bar, msg, 0.0, 0.6))
            final_audio = wav

            if keep_bg:
                msg.write("Separating background audio (Demucs)…")
                bg = P.separate_background(ss["video"], pdir)
                if bg:
                    final_audio = P.mix_background(wav, bg, bg_vol, pdir)
                else:
                    st.warning("Demucs produced no background stem - continuing without it.")
            bar.progress(0.65)

            video_for_mux = ss["video"]
            if do_lip:
                video_for_mux, skipped = P.lipsync(
                    ss["video"], wav, placements, pdir, w2l_dir, w2l_ckpt, w2l_py,
                    chunk_sec=chunk, resize_factor=int(resize),
                    progress=bar_cb(bar, msg, 0.65, 0.95))
                if skipped:
                    st.warning(f"{skipped} chunk(s) had no detectable face and were left unchanged.")

            msg.write("Muxing final video…")
            out = P.mux_video(video_for_mux, final_audio, str(pdir / f"dubbed_{ss['tgt']}.mp4"))
            bar.progress(1.0)
            msg.success("Done!")
            ss["result"] = out
        except Exception as e:
            st.error(f"Dubbing failed: {e}")

if ss.get("result") and Path(ss["result"]).exists():
    st.video(ss["result"])
    with open(ss["result"], "rb") as f:
        st.download_button("⬇️ Download dubbed video", f, file_name=Path(ss["result"]).name,
                           mime="video/mp4")
