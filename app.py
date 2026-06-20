import streamlit as st
import whisper
import os
import tempfile
import atexit
import shutil

from moviepy import VideoFileClip, AudioFileClip
from deep_translator import GoogleTranslator
from elevenlabs.client import ElevenLabs



st.title("🎬 AI English → Tamil Lip Sync Dubber")

st.write(
    "English video → Tamil voice → AI mouth sync"
)



client = ElevenLabs(

    api_key="sk_155bd03b99285b4dfecfb19008f1f347bc5e113f446329e7"

)


VOICE_ID="CwhRBWXzGAHq8TQ4Fs17"





# session


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




atexit.register(

    lambda:

    cleanup_tmpdir(

        st.session_state.get("tmpdir")

    )

)







uploaded=st.file_uploader(

    "Upload video",

    type=[

        "mp4",

        "mkv",

        "mov",

        "avi"

    ]

)








if st.button("Start AI Lip Sync Dub"):



    if uploaded:



        cleanup_tmpdir(

            st.session_state.tmpdir

        )



        tmpdir=tempfile.mkdtemp()



        st.session_state.tmpdir=tmpdir





        video_path=os.path.join(

            tmpdir,

            "input.mp4"

        )


        audio_path=os.path.join(

            tmpdir,

            "tamil_voice.mp3"

        )


        output_path=os.path.join(

            tmpdir,

            "Tamil_LipSync.mp4"

        )





        try:



            # save video


            with open(video_path,"wb") as f:

                f.write(

                    uploaded.read()

                )




            st.success(
                "Video saved"
            )





            # Whisper


            st.info(
                "Finding speech..."
            )



            model=whisper.load_model(

                "small"

            )



            result=model.transcribe(

                video_path,

                language="en",

                fp16=False

            )



            english=result["text"]



            st.write(
                english
            )





            # Translate


            st.info(
                "Translating Tamil..."
            )


            tamil=GoogleTranslator(

                source="en",

                target="ta"

            ).translate(

                english

            )





            st.write(tamil)






            # ElevenLabs


            st.info(
                "Generating Tamil voice..."
            )



            audio=client.text_to_speech.convert(


                voice_id=VOICE_ID,


                model_id="eleven_multilingual_v2",


                text=tamil


            )





            with open(audio_path,"wb") as f:


                for chunk in audio:

                    f.write(chunk)





            st.success(
                "Tamil audio ready"
            )







            # Wav2Lip


            st.info(
                "Applying AI lip sync..."
            )



            command=f"""

            python Wav2Lip/inference.py

            --checkpoint_path Wav2Lip/checkpoints/Wav2Lip_gan.pth

            --face {video_path}

            --audio {audio_path}

            --outfile {output_path}

            --pads 0 20 0 0

            --resize_factor 1

            """



            result=os.system(command)



            if result !=0:


                raise Exception(

                    "Wav2Lip failed"

                )






            st.success(
                "Lip sync completed 🎉"
            )



            st.session_state.output_path=output_path

            st.session_state.download_ready=True







        except Exception as e:


            st.error(

                str(e)

            )



            cleanup_tmpdir(tmpdir)




    else:


        st.warning(
            "Upload video first"
        )







# download


if (

st.session_state.download_ready

and st.session_state.output_path

):



    with open(

        st.session_state.output_path,

        "rb"

    ) as f:


        data=f.read()





    st.download_button(

        "⬇️ Download Tamil Lip Sync Video",

        data,

        file_name="Tamil_LipSync.mp4",

        mime="video/mp4"

    )
