# 🎙️ AI Video Dubbing Studio (Streamlit)

Dub any video into **Tamil, English, Hindi, Malayalam, Telugu, Kannada** and 20+ other languages.

Flow in the app: **Upload → choose language → Detect → Translate → Generate**

- Detects the spoken language automatically
- Separates speakers and detects **male / female** for each
- Gives every speaker their **own ElevenLabs voice**, the same voice for the whole video
- Puts each dubbed line where the original speaker started talking, and fits it into the time they used
- Optional **Wav2Lip** lip-sync (processed in chunks, so long videos work)
- Optional background music/effects kept via Demucs
- Review and edit the translation before spending voice credits
- Downloads the dubbed `.mp4` and `.srt` subtitles

## 1. Install ffmpeg (one time)

| OS | Command |
|---|---|
| Windows | `winget install ffmpeg` (then reopen the terminal) |
| Mac | `brew install ffmpeg` |
| Linux | `sudo apt install ffmpeg` |

Check: `ffmpeg -version`

## 2. Install and run

Needs Python 3.10 or newer.

```bash
python -m venv venv
venv\Scripts\activate            # Windows
source venv/bin/activate         # Mac / Linux
pip install -r requirements.txt
streamlit run app.py
```

The browser opens at http://localhost:8501.

## 3. ElevenLabs API key and voices

Create a key at elevenlabs.io → Profile → API Keys. Give it access to **Speech to Text, Text to Speech and Voices** (or leave it unrestricted).

- **Key:** paste it in the sidebar and press **Enter**, then click **Check key**. Or put it in a `.env` file (copy `.env.example`): `ELEVENLABS_API_KEY=...`
- **Voices:** the sidebar has a Male and a Female box, one voice ID per line (`id | name`). Click **Load voices from my ElevenLabs account** to fill them automatically. Defaults can also be edited in `pipeline.py` (`DEFAULT_MALE_VOICES`, `DEFAULT_FEMALE_VOICES`).
- Speaker 1 (male) gets male voice 1, speaker 2 (male) gets male voice 2, and so on. If a voice ID does not exist for your account, the app swaps in an unused voice of the same gender from your account and tells you.
- Free plans cannot use Voice Library voices through the API. Use voices from **My Voices**.

## 4. Using the app

1. **Step 1:** upload a video (or type a full file path for a very large file).
2. **Step 2:** choose the language to dub into, then click **Detect language, speakers & gender**.
3. **Step 3:** check the speaker table (fix a gender if it is wrong), then click **Translate**. You can edit any translated line.
4. **Step 4:** click **Generate dubbed video**, then download it.

Start with a 30–60 second video to test your key and voices.

## 5. Wav2Lip lip-sync (optional)

Wav2Lip needs an older Python environment, so it runs in its own virtualenv and the app calls it.

```bash
git clone https://github.com/Rudrabha/Wav2Lip.git
python3.9 -m venv wav2lip-venv
source wav2lip-venv/bin/activate          # Windows: wav2lip-venv\Scripts\activate
pip install torch torchvision             # GPU build from pytorch.org
pip install "numpy<1.24" "librosa==0.9.2" opencv-python tqdm numba scipy
deactivate
```

Download into the cloned repo (links are in the Wav2Lip README):

- `wav2lip_gan.pth` → `Wav2Lip/checkpoints/wav2lip_gan.pth`
- `s3fd.pth` → `Wav2Lip/face_detection/detection/sfd/s3fd.pth`

Then tick **Lip-sync with Wav2Lip** in the sidebar and set the three paths (or put them in `.env`). `WAV2LIP_PYTHON` must be the python inside `wav2lip-venv`. Wav2Lip is for research / non-commercial use.

## 6. If something does not work

Every step shows its error message on screen, with a **Technical details** box.

| Message | Fix |
|---|---|
| ffmpeg not found | Do step 1, reopen the terminal |
| ElevenLabs error 401 | Wrong key, or the key lacks a permission |
| ElevenLabs error 402 / paid plan | Use voices from My Voices, or upgrade the plan |
| Translation failed | Needs internet access to Google Translate |
| No speech was found | The video has no clear speech, or music is too loud |
| Wav2Lip not found | Check the three Wav2Lip paths in the sidebar |

Detection results are cached in `work/<video>/`, so pressing Detect again does not pay for transcription twice. Generated voice lines are cached too, so re-generating after editing a line only pays for that line.

## 7. Limits

- Wav2Lip syncs one face; chunks with no detectable face stay unchanged.
- Translated speech can be longer than the original. The app borrows the pause after a line and speeds up to the slider limit; for lines still too long, shorten the translated text.
- Gender comes from voice pitch. Children, very high male voices and low female voices can be misjudged, so correct them in the speaker table.
- Google Translate is machine translation; review important scripts.
- Overlapping speakers and loud music reduce speaker separation accuracy.
- Only dub content you have rights to, and tell viewers when speech is AI generated.

## Files

```
app.py            Streamlit UI
pipeline.py       all processing steps
requirements.txt  Python packages
.env.example      settings template (copy to .env)
.streamlit/config.toml   upload limit (5 GB)
```
