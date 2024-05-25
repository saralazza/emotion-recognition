
from dataset import load_dataset
from video_features import load_video

import numpy as np
from sklearn.model_selection import train_test_split


def main():

    paths, labels_name = load_dataset()

    NUM_CLASSES = len(labels_name)

    frames_list, labels = load_video(paths, labels_name)

    x_train, x_test, y_train, y_test=train_test_split(frames_list, labels, test_size=0.06, random_state=10)
    print(x_train.shape, x_test.shape, np.array(y_train).shape, np.array(y_test).shape)

if __name__ == "__main__":
    main()