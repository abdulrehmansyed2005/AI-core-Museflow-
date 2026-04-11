import os
from basic_pitch.inference import predict_and_save
from basic_pitch import ICASSP_2022_MODEL_PATH

# Make sure this matches your audio file exactly!
input_audio_path = "hum.wav" 
output_directory = "." 

if not os.path.exists(input_audio_path):
    print(f"Error: I cannot find {input_audio_path} in your folder!")
else:
    print(f"Loading {input_audio_path} into the AI...")
    predict_and_save(
        audio_path_list=[input_audio_path],
        output_directory=output_directory,
        save_midi=True,
        sonify_midi=False,
        save_model_outputs=False,
        save_notes=False,
        model_or_model_path=ICASSP_2022_MODEL_PATH
    )
    print("Done! Look in your folder on the left for the new .mid file.")