import os
import librosa
from moviepy.editor import VideoFileClip

EMOTIONS_TO_IDX = {'ha': 0, 'sa': 1, 'an': 2, 'fe': 3, 'di': 4, 'su': 5}
    

def extract_audio_from_videos(files, directories, output_directory):
    out_paths = []
    rye = output_directory + "/" + "ryerson"
    rav = output_directory + "/" + "ravdess"

    os.makedirs(output_directory, exist_ok=True)
    os.makedirs(rye, exist_ok=True)
    os.makedirs(rav, exist_ok=True)

    for i, video_path in enumerate(files):
        filename = video_path.split("/")[-1]

        if directories[i] == "RAVDESS":

            audio_path = os.path.join(rav, filename.split(".")[0] + "_" + str(i) + ".wav")
        else:
            audio_path = os.path.join(rye, filename.split(".")[0] + "_" + str(i) + ".wav")

        out_paths.append(audio_path)
        try:
            video_clip = VideoFileClip(video_path)
            audio_clip = video_clip.audio
            audio_clip.write_audiofile(audio_path, codec='pcm_s16le', logger = None)
            audio_clip.close()
            video_clip.close()
        except Exception as e:
            print(f"Failed to process {video_path}: {e}")
    return out_paths


def load_audio(file_path, sr=22050):
    dataset = file_path.split("/")[-2]
    if dataset == "ravdess":
        offset = 1.8
        duration = 2
    elif dataset == "ryerson":
        offset = 2
        duration = 2
    try:
        y, sr = librosa.load(file_path, sr=sr, duration=duration, offset=offset)
        return y, sr
    except Exception as e:
        print("Error encountered while parsing file: ", file_path)
        return None, None
    

def load_audio_dataset(files):
    labels = []
    for file in files:
        label_name = file.split("/")[-2]
        label = EMOTIONS_TO_IDX[label_name]
        labels.append(label)
    return labels