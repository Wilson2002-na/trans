import streamlit as st
import whisper
import os
import tempfile
import atexit
import shutil

from moviepy import (
    VideoFileClip,
    AudioFileClip,
    CompositeAudioClip
)

from deep_translator import GoogleTranslator
from elevenlabs.client import ElevenLabs



st.title("🎬 AI English → Tamil Video Dubber")
st.write("Automatic speaker timing + AI Tamil dubbing")



# ElevenLabs

client = ElevenLabs(

    api_key="sk_155bd03b99285b4dfecfb19008f1f347bc5e113f446329e7"

)


VOICE_ID = "CwhRBWXzGAHq8TQ4Fs17"





# Session state

if "tmpdir" not in st.session_state:

    st.session_state.tmpdir=None


if "output_path" not in st.session_state:

    st.session_state.output_path=None


if "download_ready" not in st.session_state:

    st.session_state.download_ready=False






def cleanup_tmpdir(path):

    if path and os.path.isdir(path):

        shutil.rmtree(
            path,
            ignore_errors=True
        )





def cleanup():

    cleanup_tmpdir(

        st.session_state.get("tmpdir")

    )


atexit.register(cleanup)







uploaded = st.file_uploader(

    "Upload Video",

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



        cleanup_tmpdir(

            st.session_state.tmpdir

        )



        tmpdir=tempfile.mkdtemp()


        st.session_state.tmpdir=tmpdir





        ext=os.path.splitext(

            uploaded.name

        )[1]



        video_path=os.path.join(

            tmpdir,

            "video"+ext

        )



        output_path=os.path.join(

            tmpdir,

            "Tamil_Dubbed.mp4"

        )



        final_audio_path=os.path.join(

            tmpdir,

            "tamil_audio.mp3"

        )




        try:



            # save video


            with open(video_path,"wb") as f:

                f.write(
                    uploaded.read()
                )



            st.success(
                "Video uploaded"
            )






            # Whisper


            st.info(
                "Detecting speaker timing..."
            )



            model=whisper.load_model(

                "small"

            )



            result=model.transcribe(

                video_path,

                language="en",

                fp16=False

            )



            segments=result["segments"]



            st.success(
                "Timing detected"
            )








            # Create audio clips list


            audio_clips=[]






            # Process each sentence


            for i,seg in enumerate(segments):



                start=seg["start"]

                end=seg["end"]


                text=seg["text"]


                duration=end-start



                st.write(

                    f"Processing {i+1}/{len(segments)}"

                )





                tamil=GoogleTranslator(

                    source="en",

                    target="ta"

                ).translate(

                    text

                )








                # ElevenLabs


                audio=client.text_to_speech.convert(



                    voice_id=VOICE_ID,


                    model_id="eleven_multilingual_v2",


                    text=tamil



                )







                segment_file=os.path.join(

                    tmpdir,

                    f"audio_{i}.mp3"

                )





                with open(segment_file,"wb") as f:


                    for chunk in audio:

                        f.write(chunk)







                clip=AudioFileClip(

                    segment_file

                )





                # automatic timing placement


                clip=clip.with_start(

                    start

                )





                # fit duration


                if clip.duration > duration:


                    clip=clip.subclipped(

                        0,

                        duration

                    )




                audio_clips.append(

                    clip

                )








            # combine all speech clips


            final_audio=CompositeAudioClip(

                audio_clips

            )



            final_audio.write_audiofile(

                final_audio_path,

                codec="mp3",

                logger=None

            )







            # merge with video


            video=VideoFileClip(

                video_path

            )


            audio=AudioFileClip(

                final_audio_path

            )



            final=video.with_audio(

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



            st.session_state.output_path=output_path


            st.session_state.download_ready=True



            st.success(

                "🎉 Tamil Dubbed Video Ready"

            )






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







# Download


if (

    st.session_state.download_ready

    and st.session_state.output_path

):



    with open(

        st.session_state.output_path,

        "rb"

    ) as f:


        video=f.read()





    st.download_button(

        "⬇️ Download Tamil Dubbed Video",

        video,

        file_name="Tamil_Dubbed.mp4",

        mime="video/mp4"

    )
