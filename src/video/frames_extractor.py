import cv2
import numpy as np
from tqdm import tqdm
from src.video.features_extractor import feature_from_Video
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def frames_extraction(video_path):
    """extracts max 50 frames from the video_path

    Args:
        video_path: path of the video to framerize

    Returns:
        returns (list) list of frame shape: 50x240x320x3
    """
    width = 240
    height = 320
    sequence_length = 50
    frames = []

    video_reader = cv2.VideoCapture(video_path)
    frame_count=int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    skip_interval = max(int(frame_count/sequence_length), 1)

    for counter in range(sequence_length):
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, counter * skip_interval)
        ret, frame = video_reader.read
        if not ret:
            break
        frame = cv2.resize(frame, (height, width))
        frames.append(frame)

    video_reader.release()

    return frames


def load_video(paths, label_dict):
    """extracts 50 frames from videos in paths

    Args:
        paths: list of video paths in the datasets
        labels_name: name of label

    Returns:
        returns (np.array) shape: len(paths)x50 list of video frame, 
        (np.array) list with associated labels
    """
    frames_list = []
    labels = []

    for video_path in tqdm(paths, desc = "loading video"):
        
        frames = frames_extraction(video_path)
        frames_list.append(frames)
        
        if label_dict is None:
            label = video_path.split('/')[-1][6:8]
            labels.append(int(label))
        else:
            label = video_path.split('/')[-1][:2]
            labels.append(label_dict[label])
    
#                           prima era 'float32'
    return np.array(frames_list, dtype='ubyte'), np.array(labels, dtype='int8')

def load_featured_video(paths, dir_path, label_dict):
    """extracts all features from video in paths

    Args:
        paths: list of video paths in the datasets
        labels_name: name of label

    Returns:
        returns (np.array) list of featured video, 
        (np.array) list with associated labels
    """
    features_list = []
    labels = []
    
    #WARNING: make sure the path is the correct one: 
    # - use "/kaggle/input/landmarker/face_landmarker_v2_with_blendshapes.task" for kaggle
    # - use "../../assets/face_landmarker_v2_with_blendshapes.task" from the repository 
    base_options = python.BaseOptions(model_asset_path='../../assets/face_landmarker_v2_with_blendshapes.task')
    options = vision.FaceLandmarkerOptions(base_options=base_options,
                                       output_face_blendshapes=True,
                                       output_facial_transformation_matrixes=True,
                                       num_faces=1)
    detector = vision.FaceLandmarker.create_from_options(options)
    i = 0
    for video_path in tqdm(paths, desc = "loading video"):

        frames = frames_extraction(video_path)
        features = feature_from_Video(detector,frames)
        features_list.append(features)

        if dir_path[i] == 'RAVDESS':
            label = video_path.split('/')[-1][6:8]
            labels.append(int(label)-3)
        else:
            label = video_path.split('/')[-1][:2]
            labels.append(label_dict[label])

        i += 1

    return np.array(features_list, dtype='float32'), np.array(labels, dtype='int8')
