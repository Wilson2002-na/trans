# Multilingual AI Video Dubber (Streamlit)

A starter application that uploads a video, detects its spoken language, transcribes speech with timestamps, optionally identifies speakers, translates each speech segment, generates a consistent ElevenLabs voice for each speaker, and places the generated audio at the original speech timestamps.

## Features

- Video upload: MP4, MOV, MKV, AVI, WEBM, M4V (FFmpeg-supported inputs).
- Automatic source-language detection and transcription using Faster-Whisper.
- Optional multi-speaker diarization using `pyannote.audio`.
- Segment-level translation with editable text.
- Per-speaker ElevenLabs voice assignment; the same speaker uses the same voice throughout.
- Silence and pauses are preserved by placing each generated utterance at its detected start time.
- Optional original-audio background mix.
- Optional Wav2Lip post-processing.

Supported target-language options in the UI include Tamil, English, Hindi, Malayalam, Telugu, Kannada, Bengali, Marathi, Gujarati, Punjabi, Spanish, French, German, Japanese, Korean, Arabic, Portuguese, Russian and Simplified Chinese. Actual translation/TTS quality and language support depend on the selected service and account/model.

## Important limitations (read before use)

1. **This is a working starter, not a production dubbing studio.** Long videos can require a lot of CPU/GPU time and ElevenLabs credits. Process shorter clips first.
2. **Speaker diarization is not gender detection.** The app identifies speaker IDs (SPEAKER_00, SPEAKER_01, ...). Choose a suitable voice manually for each speaker. Pitch is not a reliable way to determine gender, and the app does not make a gender claim.
3. **Voice identity is not cloned automatically.** Each speaker is mapped to a selected voice from your ElevenLabs account. Use only voices you have rights/permission to use.
4. **Timing:** the app keeps the original segment start/end positions and tries to fit generated audio into each original speaking interval. If a translation is too long, it stops rather than silently creating badly overlapping dialogue. Review translations and shorten long lines if necessary.
5. **Lip sync:** Wav2Lip is optional and is best for one visible, mostly frontal face. It does not reliably solve multi-person scenes, off-screen dialogue, profile faces, occlusion, or scenes with rapid cuts. Wav2Lip is a separate research project with its own dependencies/checkpoint and licensing conditions. Check its current license and terms before commercial use.
6. **Translation:** `deep-translator` uses Google Translate through an unofficial interface. It may be rate-limited and is not a guaranteed production API. Replace it with a paid translation API for commercial/reliable use.
7. **ElevenLabs language codes:** the example uses the multilingual model. Not every listed language is necessarily available for every voice/model/account. Check the current ElevenLabs model and language documentation.
8. **Original audio:** by default the original audio is replaced. The optional background mix also retains the original dialogue at a low volume; it cannot isolate music/effects from speech.

## Requirements

- Python 3.10 or 3.11 recommended.
- FFmpeg installed and available on PATH.
- ElevenLabs API key with text-to-speech access.
- Hugging Face account/token for optional diarization.
- Internet connection for model downloads, translation, and ElevenLabs API.
- GPU is helpful but not required for Whisper. CPU processing can be slow.

## Installation (macOS)

```bash
# 1. Install FFmpeg
brew install ffmpeg

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install Python packages
python -m pip install --upgrade pip
pip install -r requirements.txt

# Optional: speaker diarization
pip install -r requirements-diarization.txt
```

On Ubuntu/Debian, install FFmpeg with `sudo apt install ffmpeg`. On Windows, install FFmpeg and add its `bin` folder to PATH.

## Configure API keys

Copy `.env.example` to `.env` and add your own credentials:

```bash
cp .env.example .env
```

- `ELEVENLABS_API_KEY`: create/manage at https://elevenlabs.io/app/settings/api-keys
- `HF_TOKEN`: create at https://huggingface.co/settings/tokens

For diarization, open and accept the model conditions for:
- https://huggingface.co/pyannote/speaker-diarization-community-1

Keep `.env` private. Never commit API keys to GitHub.

## Run

```bash
streamlit run app.py
```

Open the local URL printed by Streamlit (normally http://localhost:8501).

## Wav2Lip setup (optional)

Wav2Lip has a separate environment/dependency stack. Do not blindly install its old requirements into the same environment if they downgrade or conflict with Streamlit/PyTorch packages.

1. Read the upstream instructions: https://github.com/Rudrabha/Wav2Lip
2. Clone the repository:

```bash
git clone https://github.com/Rudrabha/Wav2Lip.git
```

3. Download a compatible pretrained checkpoint only from a legitimate source permitted by its license and place it under `Wav2Lip/checkpoints/`.
4. Install Wav2Lip's dependencies in a separate environment if needed.
5. Set `WAV2LIP_DIR` and `WAV2LIP_CHECKPOINT` in `.env`, or enter the paths in the Streamlit UI.

The Wav2Lip button is optional. The main dubbing workflow works without it.

## How the pipeline works

1. FFmpeg extracts mono 16 kHz audio.
2. Faster-Whisper detects the source language and returns timestamped speech segments.
3. pyannote assigns speaker IDs to time ranges. The app assigns each transcript segment to the speaker with the greatest time overlap.
4. Each segment is translated separately; the user can edit translations.
5. The app sends each translated utterance to ElevenLabs using the voice selected for that speaker.
6. Each generated clip is placed at the source segment's start time. Gaps between segments remain silent.
7. FFmpeg muxes the new audio with the original video.

## Troubleshooting

- **`ffmpeg not found`:** install FFmpeg and restart the terminal/IDE.
- **Hugging Face 401/403:** accept the model conditions, create a read token, and verify `HF_TOKEN`.
- **ElevenLabs 401/403:** verify API key, account access, voice ID and usage quota.
- **Out of memory / very slow:** use Whisper `small`, shorter clips, or a machine with a supported GPU.
- **Poor speaker labels:** overlapping voices, background music, noisy audio and very short turns can confuse diarization. Try another clip or correct the speaker assignments in a future UI enhancement.
- **Long translated lines:** shorten the translation so it fits the original time slot, or implement a more advanced timeline editor.
- **Wav2Lip fails:** install its own dependencies/checkpoint in a compatible environment and review upstream setup notes.

## Security and rights

Use videos, voices, and face/lip-sync processing only when you have the necessary rights and consent. Do not impersonate people or create deceptive media. Store API credentials in environment variables, not source code.


### Manual Wav2Lip inference command

After generating `dubbed.wav`, run from the Wav2Lip repository:

```bash
python inference.py \
  --checkpoint_path checkpoints/wav2lip_gan.pth \
  --face /absolute/path/to/input_video.mp4 \
  --audio /absolute/path/to/dubbed.wav \
  --outfile /absolute/path/to/lipsynced_output.mp4
```

The Streamlit UI currently exposes the Wav2Lip integration point but reports that the audio path must be wired in for the one-click workflow. The command above is the supported manual route.
