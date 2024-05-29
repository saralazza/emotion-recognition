import cv2
import numpy as np
from tqdm import tqdm

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
        ret, frame = video_reader.read()
        # Se l'operazione non è andata a buon fine
        if not ret:
            break
        frame=cv2.resize(frame, (height, width))
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(frame)
    
    # Rilascio la risorsa
    video_reader.release()

    return frames


def load_video(paths, labels_name):
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
        label = video_path.split('/')[-1][:2]
        labels.append(labels_name.index(label))
    
#                           prima era 'float32'
    return np.array(frames_list, dtype='ubyte'), np.array(labels, dtype='int8')


