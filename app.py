"""AI Video Dubbing Studio - Streamlit UI.   Run:  streamlit run app.py"""
import hashlib
import os
import shutil
import traceback
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

ss = st.session_state
ss.setdefault("api_key", os.getenv("ELEVENLABS_API_KEY", "").strip())
ss.setdefault("male_pool", "\n".join(P.DEFAULT_MALE_VOICES))
ss.setdefault("female_pool", "\n".join(P.DEFAULT_FEMALE_VOICES))
ss.setdefault("ver", 0)          # bumps whenever the script table is replaced


def key() -> str:
    return (ss.get("api_key") or "").strip()


def _load_account_voices():
    try:
        m, f = P.list_account_voices(key())
        if m:
            ss["male_pool"] = "\n".join(m)
        if f:
            ss["female_pool"] = "\n".join(f)
        ss["voice_msg"] = f"Loaded {len(m)} male and {len(f)} female voices from your account."
    except Exception as e:
        ss["voice_msg"] = f"Could not load voices: {e}"


def progress_cb(bar, text_box, lo, hi):
    def cb(frac, text):
        bar.progress(float(min(max(lo + (hi - lo) * frac, 0.0), 1.0)))
        text_box.write(text)
    return cb


# =============================================================== sidebar
with st.sidebar:
    st.header("🔑 ElevenLabs API key")
    st.text_input("Paste your API key, then press Enter", type="password", key="api_key")
    if st.button("Check key"):
        try:
            st.success(P.check_api_key(key()))
        except Exception as e:
            st.error(str(e))

    st.header("🗣️ Voices")
    st.caption("One voice ID per line (`id | name`). Each speaker gets their own voice from the pool "
               "that matches their gender and keeps it for the whole video.")
    st.button("Load voices from my ElevenLabs account", on_click=_load_account_voices)
    if ss.get("voice_msg"):
        st.caption(ss["voice_msg"])
    st.text_area("Male voices", key="male_pool", height=120)
    st.text_area("Female voices", key="female_pool", height=120)
    tts_choice = st.selectbox("Voice model", ["Auto", "eleven_multilingual_v2", "eleven_v3",
                                              "eleven_turbo_v2_5"],
                              help="Auto: multilingual_v2 where supported, eleven_v3 for Malayalam, "
                                   "Telugu, Kannada, Bengali, Marathi etc.")

    st.header("🎬 Options")
    n_spk = st.number_input("Number of speakers (0 = detect automatically)", 0, 32, 0)
    max_speed = st.slider("Max speed-up to fit timing", 1.0, 2.0, 1.5, 0.05)
    keep_bg = st.checkbox("Keep background music/effects (needs Demucs)")
    bg_vol = st.slider("Background volume", 0.1, 1.5, 0.8, 0.05)
    do_lip = st.checkbox("Lip-sync with Wav2Lip (needs GPU + setup, see README)")
    w2l_dir = st.text_input("Wav2Lip folder", os.getenv("WAV2LIP_DIR", "Wav2Lip"))
    w2l_ckpt = st.text_input("Wav2Lip checkpoint", os.getenv("WAV2LIP_CHECKPOINT",
                                                              "Wav2Lip/checkpoints/wav2lip_gan.pth"))
    w2l_py = st.text_input("Python of the Wav2Lip environment", os.getenv("WAV2LIP_PYTHON", "python"))
    chunk = st.slider("Lip-sync chunk length (s)", 20, 180, 60)
    resize = st.selectbox("Wav2Lip resize factor", [1, 2, 4])

# =============================================================== header
st.title("🎙️ AI Video Dubbing Studio")
st.caption("Upload a video → choose a language → Detect → Translate → Generate. "
           "Works with any spoken language and any video length.")

if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
    st.error("ffmpeg is not installed (or not on PATH). Install it first - see README step 1.")
if not key():
    st.warning("Paste your ElevenLabs API key in the sidebar (then press Enter).")

# =============================================================== step 1: video
st.subheader("Step 1 · Upload your video")
up = st.file_uploader("Video file", type=["mp4", "mov", "mkv", "avi", "webm", "m4v"])
path_in = st.text_input("…or full path of a (very long) video file on this computer")

video_path, pdir = None, None
if up is not None:
    pid = hashlib.md5(f"{up.name}{up.size}".encode()).hexdigest()[:10]
    pdir = WORK / pid
    pdir.mkdir(exist_ok=True)
    video_path = pdir / up.name
    if not video_path.exists():
        video_path.write_bytes(up.getbuffer())
elif path_in.strip():
    p = Path(path_in.strip().strip('"'))
    if p.is_file():
        pid = hashlib.md5(f"{p.name}{p.stat().st_size}".encode()).hexdigest()[:10]
        pdir = WORK / pid
        pdir.mkdir(exist_ok=True)
        video_path = p
    else:
        st.error(f"File not found: {p}")
if video_path:
    st.video(str(video_path))

# =============================================================== step 2: language + detect
st.subheader("Step 2 · Choose the language and detect")
target = st.selectbox("Dub / translate into", list(P.LANGUAGES), index=1, key="target")

if st.button("🔍 Detect language, speakers & gender", type="primary"):
    if not video_path:
        st.error("Upload a video first (Step 1).")
    elif not key():
        st.error("Paste your ElevenLabs API key in the sidebar first.")
    else:
        detected = False
        with st.status("Detecting…", expanded=True) as status:
            try:
                bar, box = st.progress(0.0), st.empty()
                box.write("Extracting audio…")
                audio = P.extract_audio(str(video_path), pdir)
                duration = P.media_duration(str(video_path))
                bar.progress(0.05)

                box.write("Detecting language and speakers with ElevenLabs Scribe "
                          "(can take a few minutes for long videos)…")
                tr = P.transcribe(key(), audio, int(n_spk) or None,
                                  cache_path=pdir / f"scribe_{int(n_spk)}.json")
                segs = P.build_segments(tr["words"])
                if not segs:
                    raise RuntimeError("No speech was found in this video.")
                bar.progress(0.6)

                spk_rows = P.detect_genders(audio, segs, progress=progress_cb(bar, box, 0.6, 1.0))
                for s in segs:
                    s["translated"] = ""
                for k in ("result", "tgt", "flash"):
                    ss.pop(k, None)
                ss.update(pdir=str(pdir), video=str(video_path), duration=duration,
                          src_lang=tr["language"], segs=pd.DataFrame(segs),
                          speakers=pd.DataFrame(spk_rows))
                ss["ver"] += 1
                bar.progress(1.0)
                status.update(label="Detection finished", state="complete")
                ss["flash"] = (f"Detected language: **{P.language_label(tr['language'])}** · "
                               f"{len(spk_rows)} speaker(s) · {len(segs)} sentences. "
                               f"Now click **Translate** below.")
                detected = True
            except Exception as e:
                status.update(label="Detection failed", state="error")
                st.error(str(e))
                with st.expander("Technical details"):
                    st.code(traceback.format_exc())
        if detected:
            st.rerun()

# =============================================================== step 3: translate
if "segs" in ss:
    st.subheader("Step 3 · Check the speakers, then translate")
    if ss.get("flash"):
        st.success(ss["flash"])

    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown("**Speakers**")
        spk_df = st.data_editor(
            ss["speakers"], hide_index=True, key=f"spk_{ss['ver']}",
            disabled=["speaker", "median_f0_hz", "speaking_seconds"],
            column_config={
                "gender": st.column_config.SelectboxColumn("gender", options=["Male", "Female"]),
                "voice_override": st.column_config.TextColumn("voice ID (optional)"),
            })
        st.caption("Wrong gender? Change it here. Paste a voice ID to pin a speaker to one voice.")
    with c2:
        tgt_now = ss.get("tgt")
        st.markdown(f"**Script** · spoken: {P.language_label(ss['src_lang'])}"
                    + (f" → dubbed: {tgt_now}" if tgt_now else " · not translated yet"))
        seg_df = st.data_editor(
            ss["segs"], hide_index=True, height=340, key=f"segs_{ss['ver']}",
            column_config={
                "start": st.column_config.NumberColumn("start (s)", format="%.2f"),
                "end": st.column_config.NumberColumn("end (s)", format="%.2f"),
                "speaker": st.column_config.TextColumn("speaker", disabled=True),
                "text": st.column_config.TextColumn("original", disabled=True, width="medium"),
                "translated": st.column_config.TextColumn("dubbed text (editable)", width="large"),
            })

    if st.button(f"🌐 Translate to {target}", type="primary"):
        translated = False
        with st.status(f"Translating to {target}…", expanded=True) as status:
            try:
                bar, box = st.progress(0.0), st.empty()
                rows, failed = P.translate_segments(seg_df.to_dict("records"), target, ss["src_lang"],
                                                    progress_cb(bar, box, 0.0, 1.0))
                ss["segs"], ss["tgt"] = pd.DataFrame(rows), target
                ss["ver"] += 1
                status.update(label="Translation finished", state="complete")
                ss["flash"] = (f"Translated {len(rows) - failed}/{len(rows)} sentences to {target}. "
                               "Edit any line if you like, then click **Generate dubbed video**."
                               + (f" ⚠️ {failed} sentence(s) could not be translated and keep the original text."
                                  if failed else ""))
                translated = True
            except Exception as e:
                status.update(label="Translation failed", state="error")
                st.error(str(e))
                with st.expander("Technical details"):
                    st.code(traceback.format_exc())
        if translated:
            st.rerun()

    st.download_button("Download subtitles (.srt)", P.segments_to_srt(seg_df.to_dict("records")),
                       "dubbed_subtitles.srt")

    # =========================================================== step 4: generate
    st.subheader("Step 4 · Generate the dubbed video")
    if st.button("🎬 Generate dubbed video", type="primary"):
        if not key():
            st.error("Paste your ElevenLabs API key in the sidebar first.")
        else:
            with st.status("Generating dubbed video…", expanded=True) as status:
                try:
                    bar, box = st.progress(0.0), st.empty()
                    pdir_run = Path(ss["pdir"])
                    rows = seg_df.to_dict("records")

                    if ss.get("tgt") != target or not any(str(r.get("translated") or "").strip() for r in rows):
                        box.write(f"Translating to {target} first…")
                        rows, failed = P.translate_segments(rows, target, ss["src_lang"])
                        ss["tgt"] = target
                        if failed:
                            st.warning(f"{failed} sentence(s) could not be translated.")

                    speakers = spk_df.to_dict("records")
                    voice_map, warns = P.assign_voices(speakers, P.parse_pool(ss["male_pool"]),
                                                       P.parse_pool(ss["female_pool"]))
                    voice_map, notes = P.fix_invalid_voices(key(), voice_map, speakers)
                    for w in warns + notes:
                        st.warning(w)
                    st.write("Voices used:", pd.DataFrame(
                        [{"speaker": s["speaker"], "gender": s["gender"], "voice_id": voice_map[s["speaker"]]}
                         for s in speakers]))

                    model = P.pick_tts_model(P.LANGUAGES[target][0], tts_choice)
                    st.write(f"Voice model: `{model}`")
                    duration = ss.get("duration") or P.media_duration(ss["video"])

                    wav, placements = P.build_dub_track(key(), rows, voice_map, model, duration, pdir_run,
                                                        max_speedup=max_speed,
                                                        progress=progress_cb(bar, box, 0.0, 0.6))
                    final_audio = wav

                    if keep_bg:
                        box.write("Separating background music (Demucs)…")
                        bg = P.separate_background(ss["video"], pdir_run)
                        if bg:
                            final_audio = P.mix_background(wav, bg, bg_vol, pdir_run)
                        else:
                            st.warning("Demucs produced no background track - continuing without it.")
                    bar.progress(0.65)

                    video_for_mux = ss["video"]
                    if do_lip:
                        video_for_mux, skipped = P.lipsync(
                            ss["video"], wav, placements, pdir_run, w2l_dir, w2l_ckpt, w2l_py,
                            chunk_sec=chunk, resize_factor=int(resize),
                            progress=progress_cb(bar, box, 0.65, 0.95))
                        if skipped:
                            st.warning(f"{skipped} chunk(s) had no detectable face and were left unchanged.")

                    box.write("Creating the final video…")
                    out = P.mux_video(video_for_mux, final_audio,
                                      str(pdir_run / f"dubbed_{target}.mp4"))
                    bar.progress(1.0)
                    ss["result"] = out
                    status.update(label="Dubbed video is ready", state="complete")
                except Exception as e:
                    status.update(label="Generation failed", state="error")
                    st.error(str(e))
                    with st.expander("Technical details"):
                        st.code(traceback.format_exc())

if ss.get("result") and Path(ss["result"]).exists():
    st.success("Done! Your dubbed video:")
    st.video(ss["result"])
    with open(ss["result"], "rb") as fh:
        st.download_button("⬇️ Download dubbed video", fh, file_name=Path(ss["result"]).name,
                           mime="video/mp4")
