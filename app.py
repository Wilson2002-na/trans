# =========================
# INSTALL (run once)
# =========================
# pip install yt-dlp openai-whisper deep-translator moviepy elevenlabs

# =========================
# IMPORTS
# =========================

import yt_dlp
import whisper
from deep_translator import GoogleTranslator
from moviepy import VideoFileClip, AudioFileClip

from elevenlabs.client import ElevenLabs
from elevenlabs import save


# =========================
# 1. YouTube URL
# =========================

url = input("Paste YouTube URL: ")


# =========================
# 2. DOWNLOAD VIDEO
# =========================

print("Downloading video...")

options = {
    "format": "mp4",
    "outtmpl": "video.mp4"
}

with yt_dlp.YoutubeDL(options) as ydl:
    ydl.download([url])

print("Download completed ✅")


# =========================
# 3. SPEECH → TEXT (Whisper)
# =========================

print("Converting speech to text...")

whisper_model = whisper.load_model("large-v3")

result = whisper_model.transcribe(
    "video.mp4",
    language="en",
    fp16=False
)

segments = result["segments"]

english_sentences = []

for item in segments:
    english_sentences.append(item["text"])

print("\nEnglish Text:")
print(english_sentences)


# =========================
# 4. ENGLISH → TAMIL TRANSLATION
# =========================

print("\nTranslating to Tamil...")

translator = GoogleTranslator(source="en", target="ta")

tamil_sentences = []

for sentence in english_sentences:
    tamil = translator.translate(sentence)
    tamil_sentences.append(tamil)

tamil_text = " ".join(tamil_sentences)

print("\nTamil Text:")
print(tamil_text)


# =========================
# 5. TEXT → SPEECH (ELEVENLABS)
# =========================

print("\nCreating Tamil voice with ElevenLabs...")

client = ElevenLabs(
    api_key="sk_155bd03b99285b4dfecfb19008f1f347bc5e113f446329e7"   # 🔴 PUT YOUR KEY HERE
)

# You can change voice_id anytime
voice_id = "pNInz6obpgDQGcFmaJgB"

audio = client.text_to_speech.convert(
    voice_id=voice_id,
    text=tamil_text,
    model_id="eleven_multilingual_v2"
)

save(audio, "tamil_audio.mp3")

print("ElevenLabs audio created ✅")


# =========================
# 6. CREATE DUBBED VIDEO
# =========================

print("\nCreating Tamil dubbed video...")

video = VideoFileClip("video.mp4")
audio = AudioFileClip("tamil_audio.mp3")

final_video = video.with_audio(audio)

final_video.write_videofile(
    "Tamil_Dubbed.mp4",
    codec="libx264",
    audio_codec="aac"
)

print("\nDONE ✅")
print("Tamil Dubbed Video: Tamil_Dubbed.mp4")
