# 🎙️ AI Video Dubbing Studio (Streamlit)

Dub any video into **Tamil, English, Hindi, Malayalam, Telugu, Kannada** and 20+ more languages.

- Auto-detects the spoken language
- Separates speakers and detects **male / female** for each
- Gives every speaker their **own ElevenLabs voice** that stays the same for the whole video
- Knows **when each person starts and stops speaking** and puts the dubbed line at the same time
- Optional **Wav2Lip** lip-sync, processed in chunks so long videos work
- Optional background music/effects kept via Demucs
- Review and edit the translation and genders before spending voice credits
- Downloads dubbed `.mp4` and `.srt` subtitles

## How it works

```
video ─ffmpeg→ audio ─ElevenLabs Scribe→ words + speaker IDs + language
      → sentences (speaker, start, end) → pitch analysis → Male/Female per speaker
      → Google Translate → target language
      → ElevenLabs TTS (one fixed voice per speaker, fixed settings + seed)
      → fit each line into its original time slot (speed-up up to 1.5x, borrows the pause after it)
      → dubbed track [+ Demucs background] → Wav2Lip (chunks cut in silences) → ffmpeg mux → MP4
```

## 1. Requirements

| What | Why |
|---|---|
| Python 3.10+ | app |
| **ffmpeg** + ffprobe on PATH | audio/video processing (`sudo apt install ffmpeg` / `brew install ffmpeg` / `winget install ffmpeg`) |
| ElevenLabs API key | transcription (Scribe), speaker detection, voices (paid plan recommended for long videos) |
| NVIDIA GPU (optional) | only for Wav2Lip lip-sync |

## 2. Install

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env              # put your ELEVENLABS_API_KEY inside
streamlit run app.py
```

Optional background-music preservation:

```bash
pip install demucs torch
```

## 3. Wav2Lip setup (only if you want lip-sync)

Wav2Lip uses old libraries, so it lives in its **own virtualenv**; the app calls it as a subprocess.

```bash
git clone https://github.com/Rudrabha/Wav2Lip.git
python3.9 -m venv wav2lip-venv && source wav2lip-venv/bin/activate   # Python 3.8–3.10 works best
pip install torch torchvision                                         # GPU build from pytorch.org
pip install "numpy<1.24" "librosa==0.9.2" opencv-python tqdm numba scipy
deactivate
```

Download into the cloned repo (links are in the Wav2Lip README):

- `wav2lip_gan.pth` → `Wav2Lip/checkpoints/wav2lip_gan.pth`
- `s3fd.pth` (face detector) → `Wav2Lip/face_detection/detection/sfd/s3fd.pth`

Then set in `.env` (or in the sidebar):

```
WAV2LIP_DIR=Wav2Lip
WAV2LIP_CHECKPOINT=Wav2Lip/checkpoints/wav2lip_gan.pth
WAV2LIP_PYTHON=/full/path/to/wav2lip-venv/bin/python
```

> Wav2Lip is released for **research / non-commercial use**. Check its license before commercial use.

## 4. Using the app

1. Paste your ElevenLabs key, pick the target language.
2. Upload a video (or give a file path for very large files). Uploads up to 5 GB are enabled in `.streamlit/config.toml`.
3. Click **Detect language, speakers & gender, then translate**.
4. Review the speaker table (fix gender if wrong, or pin a voice ID) and edit any translated line.
5. Click **Generate dub**. Download the video when done.

### Voices
- The sidebar has two pools: male and female voice IDs. Defaults are ElevenLabs premade voices.
- **Load voices from my ElevenLabs account** fills the pools using the gender labels of your voices (add your own or library voices there).
- Speaker 1 (male) gets male voice 1, speaker 2 (male) gets male voice 2, and so on. A speaker keeps the same voice in every sentence. If you have more speakers than voices, voices get reused and the app warns you, so add more IDs.

### Voice models
`Auto` uses `eleven_multilingual_v2` for English, Hindi, Tamil, and other languages it supports, and `eleven_v3` for Malayalam, Telugu, Kannada, Bengali, Marathi, Gujarati, Punjabi, Urdu, Nepali, Thai, and Vietnamese. Check ElevenLabs' docs for the current language list of each model.

## 5. Long videos

- Audio is compressed before transcription, so the upload stays small.
- Lip-sync is done in ~60 s chunks cut at pauses; progress is shown per chunk.
- TTS clips are cached in `work/<video-id>/tts`, so re-running after an edit only pays for changed lines.

## 6. Limitations (please read)

- **Multiple faces:** Wav2Lip syncs one face per frame (the one its detector returns), so it works best when the person talking is the main face. Chunks without a detectable face are left unchanged.
- **Timing:** translated speech can be longer than the original. The app borrows the pause after a line and speeds up to the slider limit; beyond that, lines may overlap slightly. Edit the translation shorter for those lines.
- **Gender detection** uses pitch, so very high male voices, low female voices, and children can be misread. Correct them in the table.
- **Translation** uses Google Translate (via `deep-translator`); review the script for important content.
- **Overlapping speakers** and loud music can reduce diarization accuracy.
- Costs: Scribe and TTS both use ElevenLabs credits. Only dub content you have rights to, and tell viewers when speech is AI generated.

## Files

```
app.py            Streamlit UI
pipeline.py       all processing steps
requirements.txt  Python dependencies
.env.example      configuration template
.streamlit/config.toml   upload size limit
```
