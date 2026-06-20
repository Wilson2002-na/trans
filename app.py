import streamlit as st
import whisper
import os
import tempfile
import atexit
import shutil

from moviepy import VideoFileClip, AudioFileClip
from deep_translator import GoogleTranslator
from elevenlabs.client import ElevenLabs


st.title("🎬 AI English → Tamil Video Dubber")
st.write("Upload a video file to get a Tamil dubbed version")


# ElevenLabs API

client = ElevenLabs(
    api_key="sk_155bd03b99285b4dfecfb19008f1f347bc5e113f446329e7"
)


VOICE_ID = "CwhRBWXzGAHq8TQ4Fs17"



# ---------------- Voice Controls ----------------

st.sidebar.title("🎙️ Speaker Controls")


speech_speed = st.sidebar.slider(
    "Speaker Speed",
    0.7,
    1.3,
    1.0,
    0.1
)


stop_speaker = st.sidebar.checkbox(
    "Stop Speaker"
)



# Session state

if "tmpdir" not in st.session_state:
    st.session_state.tmpdir = None


if "output_path" not in st.session_state:
    st.session_state.output_path = None


if "download_ready" not in st.session_state:
    st.session_state.download_ready = False


if "download_clicked" not in st.session_state:
    st.session_state.download_clicked = False




def cleanup_tmpdir(path):

    if path and os.path.isdir(path):

        shutil.rmtree(
            path,
            ignore_errors=True
        )



def _atexit_cleanup():

    cleanup_tmpdir(
        st.session_state.get("tmpdir")
    )


atexit.register(
    _atexit_cleanup
)



# delete files after download

if st.session_state.download_clicked:


    cleanup_tmpdir(
        st.session_state.tmpdir
    )


    st.session_state.tmpdir = None
    st.session_state.output_path = None
    st.session_state.download_ready = False
    st.session_state.download_clicked = False


    st.success(
        "Temporary files deleted"
    )





uploaded = st.file_uploader(

    "Upload Video File",

    type=[
        "mp4",
        "mkv",
        "avi",
        "mov",
        "webm"
    ]

)





if st.button("Start Tamil Dubbing"):



    if uploaded:


        if stop_speaker:

            st.warning(
                "Speaker stopped"
            )

            st.stop()



        cleanup_tmpdir(
            st.session_state.tmpdir
        )



        tmpdir = tempfile.mkdtemp()


        st.session_state.tmpdir = tmpdir



        ext = os.path.splitext(uploaded.name)[1]



        video_path = os.path.join(
            tmpdir,
            "video"+ext
        )



        audio_path = os.path.join(
            tmpdir,
            "tamil_audio.mp3"
        )



        output_path = os.path.join(
            tmpdir,
            "Tamil_Dubbed.mp4"
        )




        try:


            # save video


            st.info(
                "Saving video..."
            )


            with open(video_path,"wb") as f:

                f.write(
                    uploaded.read()
                )



            st.success(
                "Video saved"
            )




            # whisper


            st.info(
                "Converting speech..."
            )



            model = whisper.load_model(
                "small"
            )



            result = model.transcribe(

                video_path,

                language="en",

                fp16=False

            )



            english_text = " ".join(

                [
                    x["text"]

                    for x in result["segments"]

                ]

            )



            st.success(
                "Transcript completed"
            )




            with st.expander(
                "English Transcript"
            ):

                st.write(
                    english_text
                )






            # translate


            st.info(
                "Translating English → Tamil"
            )



            tamil_text = GoogleTranslator(

                source="en",

                target="ta"

            ).translate(
                english_text
            )



            st.success(
                "Tamil translated"
            )



            with st.expander(
                "Tamil Text"
            ):

                st.write(
                    tamil_text
                )






            # ElevenLabs


            if stop_speaker:


                st.warning(
                    "Speaker stopped"
                )

                st.stop()




            st.info(

                f"Generating AI voice Speed {speech_speed}x"

            )



            audio = client.text_to_speech.convert(


                voice_id=VOICE_ID,


                model_id="eleven_multilingual_v2",


                text=tamil_text,


                voice_settings={

                    "stability":0.5,

                    "similarity_boost":0.75,

                    "style":0.3,

                    "use_speaker_boost":True,

                    "speed":speech_speed

                }


            )





            with open(audio_path,"wb") as f:


                for chunk in audio:

                    f.write(chunk)




            st.success(
                "Tamil voice created"
            )






            # merge video



            st.info(
                "Creating dubbed video..."
            )



            video = VideoFileClip(
                video_path
            )


            audio = AudioFileClip(
                audio_path
            )



            if audio.duration > video.duration:


                audio = audio.subclipped(

                    0,

                    video.duration

                )



            final = video.with_audio(
                audio
            )



            final.write_videofile(

                output_path,

                codec="libx264",

                audio_codec="aac",

                logger=None

            )




            video.close()

            audio.close()



            st.success(
                "Tamil Dubbed Video Ready 🎉"
            )



            st.session_state.output_path = output_path


            st.session_state.download_ready = True






        except Exception as e:


            st.error(
                f"Error: {e}"
            )


            cleanup_tmpdir(
                tmpdir
            )



    else:


        st.warning(
            "Upload video first"
        )






# download


if (

    st.session_state.download_ready

    and st.session_state.output_path

):


    if os.path.exists(

        st.session_state.output_path

    ):



        with open(

            st.session_state.output_path,

            "rb"

        ) as f:


            video_bytes = f.read()




        clicked = st.download_button(

            "⬇️ Download Tamil Dubbed Video",

            video_bytes,

            file_name="Tamil_Dubbed.mp4",

            mime="video/mp4"

        )



        if clicked:


            st.session_state.download_clicked = True

            st.rerun()
