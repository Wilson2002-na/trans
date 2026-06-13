from flask import Flask, render_template, request, send_file

import yt_dlp
import whisper

from gtts import gTTS
from moviepy import VideoFileClip, AudioFileClip

from deep_translator import GoogleTranslator

import os


app = Flask(__name__)


@app.route("/")
def home():

    return render_template("index.html")





@app.route("/dub", methods=["POST"])
def dub_video():


    url = request.form["url"]


    # 1. Download YouTube video

    print("Downloading...")


    options = {

        "format": "mp4",
        "outtmpl": "video.mp4",

        "http_headers": {
            "User-Agent":
            "Mozilla/5.0"
        },

        "extractor_args": {

            "youtube": {

                "player_client":
                ["android"]

            }

        }

    }



    with yt_dlp.YoutubeDL(options) as ydl:

        ydl.download([url])



    print("Downloaded")





    # 2. Whisper speech recognition


    print("Speech to text...")


    model = whisper.load_model(
        "large-v3"
    )


    result = model.transcribe(
        "video.mp4",
        language="en",
        fp16=False
    )



    segments = result["segments"]


    english = []


    for item in segments:

        english.append(
            item["text"]
        )





    # 3. Translate English to Tamil


    print("Translating...")


    translator = GoogleTranslator(

        source="en",
        target="ta"

    )



    tamil = []



    for sentence in english:


        translated = translator.translate(
            sentence
        )


        tamil.append(
            translated
        )



    tamil_text = " ".join(tamil)





    # 4. Tamil voice


    print("Creating voice...")


    voice = gTTS(

        text=tamil_text,

        lang="ta",

        slow=False

    )


    voice.save(
        "tamil_audio.mp3"
    )






    # 5. Merge audio and video


    print("Creating video...")


    video = VideoFileClip(
        "video.mp4"
    )


    audio = AudioFileClip(
        "tamil_audio.mp3"
    )



    final = video.with_audio(
        audio
    )


    final.write_videofile(

        "Tamil_Dubbed.mp4",

        codec="libx264",

        audio_codec="aac"

    )



    print("Finished")




    return send_file(

        "Tamil_Dubbed.mp4",

        as_attachment=True

    )







if __name__ == "__main__":


    app.run(

        host="0.0.0.0",

        port=5000

    )
