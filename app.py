import streamlit as st
import whisper
import os
import tempfile
import atexit
import shutil

from moviepy import VideoFileClip, AudioFileClip
from deep_translator import GoogleTranslator
from elevenlabs.client import ElevenLabs

from pydub import AudioSegment



st.title("🎬 AI English → Tamil Video Dubber")
st.write("Automatic AI dubbing with speaker timing sync")



# ElevenLabs

client = ElevenLabs(

    api_key="sk_155bd03b99285b4dfecfb19008f1f347bc5e113f446329e7"

)


VOICE_ID = "CwhRBWXzGAHq8TQ4Fs1"





# session

if "tmpdir" not in st.session_state:

    st.session_state.tmpdir=None



if "output_path" not in st.session_state:

    st.session_state.output_path=None



if "download_ready" not in st.session_state:

    st.session_state.download_ready=False




if "download_clicked" not in st.session_state:

    st.session_state.download_clicked=False







def cleanup_tmpdir(path):

    if path and os.path.isdir(path):

        shutil.rmtree(
            path,
            ignore_errors=True
        )






def exit_cleanup():

    cleanup_tmpdir(

        st.session_state.get("tmpdir")

    )



atexit.register(exit_cleanup)






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

            "final_audio.mp3"

        )






        try:



            # save video


            with open(video_path,"wb") as f:

                f.write(

                    uploaded.read()

                )



            st.success(
                "Video loaded"
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

                "Speech timing detected"

            )







            # create empty audio timeline


            video=VideoFileClip(

                video_path

            )



            final_audio=AudioSegment.silent(

                duration=int(

                    video.duration*1000

                )

            )






            # process every sentence


            for index,seg in enumerate(segments):



                start=seg["start"]

                end=seg["end"]


                duration=end-start



                english=seg["text"]



                st.write(

                    f"Processing sentence {index+1}/{len(segments)}"

                )





                # translate


                tamil=GoogleTranslator(

                    source="en",

                    target="ta"

                ).translate(

                    english

                )







                # elevenlabs


                audio=client.text_to_speech.convert(



                    voice_id=VOICE_ID,


                    model_id="eleven_multilingual_v2",


                    text=tamil



                )






                segment_file=os.path.join(

                    tmpdir,

                    f"segment_{index}.mp3"

                )




                with open(segment_file,"wb") as f:


                    for chunk in audio:

                        f.write(chunk)







                tamil_audio=AudioSegment.from_file(

                    segment_file

                )







                # AUTO LIPSYNC SPEED CONTROL



                target=int(

                    duration*1000

                )



                current=len(

                    tamil_audio

                )



                if current > target:



                    speed=current/target



                    new_rate=int(

                        tamil_audio.frame_rate*speed

                    )



                    tamil_audio=tamil_audio._spawn(

                        tamil_audio.raw_data,

                        overrides={

                            "frame_rate":new_rate

                        }

                    ).set_frame_rate(

                        tamil_audio.frame_rate

                    )








                # cut extra audio


                tamil_audio=tamil_audio[:target]





                # add silence if short


                if len(tamil_audio)<target:



                    tamil_audio += AudioSegment.silent(

                        duration=target-len(tamil_audio)

                    )






                # place at original time


                final_audio=final_audio.overlay(

                    tamil_audio,

                    position=int(start*1000)

                )








            # save final audio


            final_audio.export(

                final_audio_path,

                format="mp3"

            )








            # merge


            audio=AudioFileClip(

                final_audio_path

            )



            final_video=video.with_audio(

                audio

            )



            final_video.write_videofile(

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

                str(e)

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


    with open(

        st.session_state.output_path,

        "rb"

    ) as f:


        data=f.read()





    if st.download_button(

        "⬇️ Download Tamil Video",

        data,

        file_name="Tamil_Dubbed.mp4",

        mime="video/mp4"

    ):


        st.session_state.download_clicked=True

        st.rerun()
