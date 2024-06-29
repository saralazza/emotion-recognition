
import os
import json
import numpy as np
import joblib
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import librosa

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

from moviepy.editor import VideoFileClip

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn


def load_dataset(directory, flag=False):
    """Loads the datasets from the folders "./datasets/ryerson-emotion-database" and "./datasets/ravdess-emotional-speech-video"

    Return:
        (list) of paths of the video in the datasets,
        (list) of different labels found in the dataset
    """
    paths = []

    for subject in os.listdir(directory):

        if subject == ".DS_Store": 
            continue
        if not flag:
            percorso_dir_subject = os.path.join(directory, subject)
        else:
            percorso_dir_subject = os.path.join(directory, subject, subject)

        for elem in os.listdir(percorso_dir_subject):
            if elem == ".DS_Store": 
                continue
            percorso_elem = os.path.join(percorso_dir_subject, elem)

            if not os.path.isdir(percorso_elem):
                if not percorso_elem.endswith(".db"):
                    paths.append(percorso_elem)
            else:
                for file in os.listdir(percorso_elem):
                    percorso_file = os.path.join(percorso_elem, file)
                    if not percorso_file.endswith(".db"):
                        if not flag:
                            if file.startswith("01") and file[6:8] != "01" and file[6:8] != "02":
                                paths.append(percorso_file)
                        else:
                            paths.append(percorso_file)

    return paths


# Save video features

def save_dataset(features, labels, filepath):
    """Save the processed dataset with all features extracted

    Args:
        features: list of features for each sample
        labels: list of label for each sample
        filepath: target directory of the file
    """
    joblib.dump((features, labels), filepath)
    print(f"Dataset saved in  {filepath}")


def load_dataset_jlb(filepath):
    """Load the dataset with all features extracted from the file

    Args:
        filepath: file from which load the dataset

    Return:
        features: (list) of features for each sample
        labels: (list) of label for each sample


    """
    features, labels = joblib.load(filepath)
    print(f"Dataset loaded from {filepath}")
    return features, labels


# Extract video features

def feature_from_Video(detector, frame_video_list):
    face_blendshapes_scores_list = []
    for frame in frame_video_list :
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        detection_result = detector.detect(image)
        if not detection_result.face_blendshapes:
            if not face_blendshapes_scores_list:
                face_blendshapes_scores_list.append(np.zeros(shape=52))
            else: 
                face_blendshapes_scores_list.append(face_blendshapes_scores_list[-1])
        else:
            face_blendshapes_scores_list.append([cat.score for cat in detection_result.face_blendshapes[0]])
    return face_blendshapes_scores_list


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


def load_video(paths, dir_path, label_dict):
    """extracts 50 frames from videos in paths

    Args:
        paths: list of video paths in the datasets
        labels_name: name of label

    Returns:
        returns (np.array) shape: len(paths)x50 list of video frame,
        (np.array) list with associated labels
    """
    frames_list = []
    features_list = []
    labels = []

    base_options = python.BaseOptions(model_asset_path='/kaggle/input/landmark/face_landmarker_v2_with_blendshapes.task')
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


# Video model

class VideoEmotionRecognitionModel:
    def __init__(self, input_shape, num_classes, lstm_units=64, dense_units=32, dropout_rate=0.5, learning_rate=0.001):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(LSTM(self.lstm_units, input_shape=self.input_shape, return_sequences=True))
        model.add(Dropout(self.dropout_rate))
        model.add(LSTM(self.lstm_units))
        model.add(Dropout(self.dropout_rate))
        model.add(Dense(self.dense_units, activation='relu'))
        model.add(Dense(self.num_classes, activation='softmax'))

        model.compile(optimizer=Adam(learning_rate=self.learning_rate),
                      loss=SparseCategoricalCrossentropy(),
                      metrics=[SparseCategoricalAccuracy()])
        return model

    def summary(self):
        self.model.summary()

    def train(self, X_train, y_train,  X_val, y_val, epochs=50, batch_size=32):
        checkpoint_callback = ModelCheckpoint(
            filepath='/kaggle/working/models/video_best_model.keras',
            monitor='val_sparse_categorical_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )

        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=[checkpoint_callback]
        )

        return history

    def evaluate(self, X_test, y_test):
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test)
        return test_loss, test_accuracy

    def predict(self, X):
        predictions = self.model.predict(X)
        predicted_classes = np.argmax(predictions, axis=1)
        return predicted_classes

    def predict_prob(self, X):
        predictions = self.model.predict(X)
        predicted_classes = np.argmax(predictions, axis=1)
        return predictions, predicted_classes

    def plot_confusion_matrix(self, y_true, y_pred, class_names, title):
        cm = confusion_matrix(y_true, y_pred)
        cm = cm / np.sum(cm, axis = 1, keepdims = True)
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='.2%', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.title(title)
        plt.show()


# Train Video Model From Path

def train_video_model(train_path, train_dir_name, test_path, test_dir_name, emotions_to_idx):
    train_dataset_filepath = '/kaggle/working/Video/subtrain_features_rav.pkl'
    test_dataset_filepath = '/kaggle/working/Video/subtest_features_rav.pkl'
    os.makedirs('/kaggle/working/Video/', exist_ok = True)

    if os.path.exists(train_dataset_filepath):
        train_point_list, train_labels = load_dataset_jlb(train_dataset_filepath)
        test_point_list, test_labels = load_dataset_jlb(test_dataset_filepath)
    else:
        train_point_list, train_labels = load_video(train_path, train_dir_name, emotions_to_idx)
        save_dataset(train_point_list, train_labels, train_dataset_filepath)

        test_point_list, test_labels = load_video(test_path, test_dir_name, emotions_to_idx)
        save_dataset(test_point_list, test_labels, test_dataset_filepath)

    NUM_CLASSES = len(emotions_to_idx.keys())

    os.makedirs('/kaggle/working/models/', exist_ok = True)

    emotion_model = VideoEmotionRecognitionModel(input_shape=(50, 52), num_classes=NUM_CLASSES)

    emotion_model.summary()

    history = emotion_model.train(train_point_list, train_labels, test_point_list, test_labels, epochs=70, batch_size=32)


def video_predict(test_path, test_dir_name, features_file_name, emotions_to_idx, cm_title):

    os.makedirs('/kaggle/working/Video/', exist_ok = True)
    labels_name = ['Happy', 'Sad', 'Angry', 'Fearful', 'Disgusted', 'Surprised']

    model_path = '/kaggle/working/models/video_best_model.keras'

    if os.path.exists(features_file_name):
        test_point_list, test_labels = load_dataset_jlb(features_file_name)
    else:
        test_point_list, test_labels = load_video(test_path, test_dir_name, emotions_to_idx)
        save_dataset(test_point_list, test_labels, features_file_name)

    NUM_CLASSES = len(emotions_to_idx.keys())

    model = VideoEmotionRecognitionModel(input_shape=(50, 52), num_classes=NUM_CLASSES)
    model.model.load_weights(model_path)

    model.summary()

    test_predict_probs, test_predict = model.predict_prob(test_point_list)

    model.plot_confusion_matrix(test_labels, test_predict, labels_name, cm_title)

    return test_predict_probs, test_predict


# Audio extraction by Video

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

def extract_MFCC(file_path, n_mfcc=64):
    y, sr = load_audio(file_path)
    if y is None:
        return None
    mfcc_spectrogram = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    if mfcc_spectrogram is not None:
        if mfcc_spectrogram.shape != (64, 87):
            if mfcc_spectrogram.shape[1] > 87:
                mfcc_spectrogram = mfcc_spectrogram[:, :87]
            else:
                mfcc_spectrogram = np.pad(mfcc_spectrogram, ((0, 0), (0, 87 - mfcc_spectrogram.shape[1])), mode='constant')
    return mfcc_spectrogram


def dump_spectrogram(mel_spectrogram, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    np.save(output_file, mel_spectrogram)


def preprocess_dataset(files, output_folder):
    cont = 0
    ravdess_dict = {3: 'ha', 4: 'sa', 5: 'an', 6: 'fe', 7: 'di', 8: 'su'}
    features_path = []
    for f in files:
        directory = f.split("/")[-2]
        filename = f.split("/")[-1].split(".")[0]
        if directory == "ryerson":
                emotion = filename[:2]
                audio_file = f
                output_file = os.path.join(
                    output_folder, emotion,
                    filename + ".npy",
                )
                mfcc_spectrogram = extract_MFCC(audio_file)
                dump_spectrogram(mfcc_spectrogram, output_file)
                cont += 1

        elif directory == "ravdess": 
            part = filename.split('-')
            emotion = ravdess_dict[int(part[2])]
            audio_file = f
            output_file = os.path.join(
                output_folder, emotion,
                filename + ".npy",
            )
            mfcc_spectrogram = extract_MFCC(audio_file)
            dump_spectrogram(mfcc_spectrogram, output_file)
            cont += 1

        features_path.append(output_file)
    return features_path


# Dataset and Dataloader

class MultilabelSpectrogramDataset(Dataset):
    def __init__(self, file_paths, labels):
        self.file_paths = file_paths
        self.labels = labels

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        spectrogram = np.load(file_path)
        label = self.labels[idx]
        return torch.tensor(spectrogram, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


def load_audio_dataset(files):
    labels = []
    for file in files:
        label_name = file.split("/")[-2]
        label = emotions_to_idx[label_name]
        labels.append(label)
    return labels


# The model

class AudioModel(nn.Module):
    def __init__(self, num_labels=6):
        super(AudioModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding="same")
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding="same")
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding="same")
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding="same")

        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(128 * 4 * 5, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(0.1)
        self.fc4 = nn.Linear(128, num_labels)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.unsqueeze(1)

        x = self.relu(self.conv1(x))
        x = self.pool(x)

        x = self.relu(self.conv2(x))
        x = self.pool(x)

        x = self.relu(self.conv3(x))
        x = self.pool(x)

        x = self.relu(self.conv4(x))
        x = self.pool(x)

        x = x.flatten(start_dim=1)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc4(x)
        x = self.softmax(x)

        return x


# Train loop

def train_audio_model(subtrain_dataloader, subtest_dataloader):

    model = AudioModel()

    # Labels names for the plots
    labels_list = ['Happy', 'Sad', 'Angry', 'Fearful', 'Disgusted', 'Surprised']

    num_labels = len(labels_list)
    confusion_matrix_ = np.zeros((num_labels, num_labels))

    os.makedirs("/kaggle/working/models", exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.0001, weight_decay=0.0001)

    losses = []
    accuracy_scores = dict()

    num_epochs = 70

    for epoch in range(num_epochs):
        model.train()

        total_loss = 0.0

        for audios, labels in tqdm(subtrain_dataloader):
            audios = audios.to(device)
            labels = labels.to(device)

            outputs = model(audios)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss}')

        model.eval()

        all_preds = []
        all_labels = []

        for audios, labels in subtest_dataloader:
            audios = audios.to(device)
            labels = labels.to(device)

            outputs = model(audios)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            confusion_matrix_ += confusion_matrix(labels.cpu().numpy(), preds.cpu().numpy(), labels=range(num_labels))

        accuracy = accuracy_score(all_labels, all_preds)
        print(f'Validation accuracy: {accuracy}')

        accuracy_scores[accuracy] = model.state_dict()

        losses.append(total_loss)

    best_accuracy = max(accuracy_scores.keys())
    best_model = accuracy_scores[best_accuracy]
    print("\nBest validation accuracy: ", best_accuracy, "\n")
    torch.save(best_model, '/kaggle/working/models/audio_model.pth')
    accuracy_scores = list(accuracy_scores.keys())

    # Plot Training Loss
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Audio training Loss')
    plt.show()

    # Plot Validation Accuracy
    plt.plot(accuracy_scores)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Audio validation Accuracy')
    plt.show()

    confusion_matrix_ = confusion_matrix_ / np.sum(confusion_matrix_, axis=1, keepdims=True)

    # confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix_, annot=True, fmt=".2%", xticklabels=labels_list, yticklabels=labels_list, cmap='Purples')
    plt.title('Audio prediction on subtest set on training mode')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

def create_audio_model():

    print("Audio extraction by video of subtrain set...\n")
    audio_paths_subtrain = extract_audio_from_videos(subtrain_data, subtrain_dir, '/kaggle/working/Audios/subtrain')
    print("Audio extraction by video of subtest set...\n")
    audio_paths_subtest = extract_audio_from_videos(subtest_data, subtest_dir, '/kaggle/working/Audios/subtest')

    print("Features extraction by audio of subtrain set...\n")
    output_folder = "/kaggle/working/Dataset/subtrain"
    features_paths_subtrain = preprocess_dataset(audio_paths_subtrain, output_folder)

    print("Features extraction by audio of subtest set...\n")
    output_folder = "/kaggle/working/Dataset/subtest"
    features_paths_subtest = preprocess_dataset(audio_paths_subtest, output_folder)

    # estrazione delle label delle features
    subtrain_labels = load_audio_dataset(features_paths_subtrain)
    subtest_labels = load_audio_dataset(features_paths_subtest)

    BATCH_SIZE = 32

    subtrain_dataset = MultilabelSpectrogramDataset(features_paths_subtrain, subtrain_labels)

    subtest_dataset = MultilabelSpectrogramDataset(features_paths_subtest, subtest_labels)

    subtrain_dataloader = DataLoader(subtrain_dataset, batch_size=BATCH_SIZE, shuffle=True)
    subtest_dataloader = DataLoader(subtest_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print("Training audio model...\n")
    train_audio_model(subtrain_dataloader, subtest_dataloader)

    return subtest_dataloader, subtest_labels


def audio_predict(subtest_dataloader_audio):
    # audio, fetature and label extraction by videos

    print("Audio extraction by video of test set...\n")
    audio_paths_test = extract_audio_from_videos(test_data, test_dir, '/kaggle/working/Audios/test')

    print("Features extraction by audio of test set...\n")
    output_folder = "/kaggle/working/Dataset/test"
    features_paths_test = preprocess_dataset(audio_paths_test, output_folder)

    # label extraction by test features
    test_labels = load_audio_dataset(features_paths_test)

    BATCH_SIZE = 32

    #  dataset creation with [spectogram, label]
    test_dataset = MultilabelSpectrogramDataset(features_paths_test, test_labels)

    # converting dataset in dataloader
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print("Loading of audio model by file...\n")
    model_path = '/kaggle/working/models/audio_model.pth'

    model = AudioModel()

    model.load_state_dict(torch.load(model_path))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    model.eval()

    labels_list = ['Happy', 'Sad', 'Angry', 'Fearful', 'Disgusted', 'Surprised']
    num_labels = len(labels_list)
    confusion_matrix_ = np.zeros((num_labels, num_labels))
    subtest_preds = []
    all_labels = []
    all_outputs_subtest = []

    for audios, labels in subtest_dataloader_audio:
        audios = audios.to(device)
        labels = labels.to(device)

        outputs = model(audios)
        all_outputs_subtest.extend(outputs.cpu().detach().numpy())
        _, preds = torch.max(outputs, 1)

        subtest_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        confusion_matrix_ += confusion_matrix(labels.cpu().numpy(), preds.cpu().numpy(), labels=range(num_labels))

    accuracy = accuracy_score(all_labels, subtest_preds)
    print(f'\n\nAccuracy of audio model on subtest set: {accuracy}\n')

    confusion_matrix_ = confusion_matrix_ / np.sum(confusion_matrix_, axis=1, keepdims=True)

    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix_, annot=True, fmt=".2%", xticklabels=labels_list, yticklabels=labels_list, cmap = "Purples")
    plt.title('Audio prediction on subtest set')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

    confusion_matrix_ = np.zeros((num_labels, num_labels))
    test_preds = []
    all_labels = []
    all_outputs_test = []
    for audios, labels in test_dataloader:
        audios = audios.to(device)
        labels = labels.to(device)

        outputs = model(audios)
        _, preds = torch.max(outputs, 1)

        all_outputs_test.extend(outputs.cpu().detach().numpy())
        test_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        confusion_matrix_ += confusion_matrix(labels.cpu().numpy(), preds.cpu().numpy(), labels=range(num_labels))

    accuracy = accuracy_score(all_labels, test_preds)
    print(f'\n\nAccuracy of audio model on test set: {accuracy}\n')

    confusion_matrix_ = confusion_matrix_ / np.sum(confusion_matrix_, axis=1, keepdims=True)

    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix_, annot=True, fmt=".2%", xticklabels=labels_list, yticklabels=labels_list, cmap='Purples')
    plt.title('Audio prediction on test set')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

    return subtest_preds, test_preds, test_labels, all_outputs_subtest, all_outputs_test


# Terzo Modello

class VideoAudioDataset(Dataset):
    def __init__(self, preds, labels):
        self.preds = preds
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        pred = self.preds[idx]
        label = self.labels[idx]
        return torch.tensor(pred, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


class VideoAudioModel(nn.Module):
    def __init__(self, num_labels=6):
        super(VideoAudioModel, self).__init__()
        self.fc1 = nn.Linear(12,128)
        self.fc2 = nn.Linear(128,256)
        self.fc3 = nn.Linear(256,512)
        self.fc4 = nn.Linear(512,128)
        self.fc5 = nn.Linear(128,64)
        self.fc6 = nn.Linear(64,num_labels)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.dropout(torch.relu(self.fc2(x)))
        x = self.dropout(torch.relu(self.fc3(x)))
        x = self.dropout(torch.relu(self.fc4(x)))
        x = self.dropout(torch.relu(self.fc5(x)))
        x = self.fc6(x)
        return x

    def train_model(self, train_dataloader, test_dataloader):
        labels_list = ['Happy', 'Sad', 'Angry', 'Fearful', 'Disgusted', 'Surprised']
        num_labels = len(labels_list)
        confusion_matrix_ = np.zeros((num_labels, num_labels))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        losses = []
        accuracy_scores = dict()

        num_epochs = 100

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            self.parameters(), lr=0.0001, weight_decay=0.0001)

        last_model = self.state_dict()

        for epoch in range(num_epochs):
            self.train()
            total_loss = 0.0
            for inputs, labels in tqdm(train_dataloader):
                pred = inputs.to(device)
                labels = labels.to(device)

                outputs = self(pred)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss}')

            self.eval()

            all_preds = []
            all_labels = []

            for inputs, labels in test_dataloader:
                pred = inputs.to(device)
                labels = labels.to(device)

                outputs = self(pred)
                _, preds = torch.max(outputs, 1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                confusion_matrix_ += confusion_matrix(labels.cpu().numpy(), preds.cpu().numpy(), labels=range(num_labels))

            accuracy = accuracy_score(all_labels, all_preds)
            print(f'Validation accuracy: {accuracy}')

            accuracy_scores[accuracy] = self.state_dict()

            losses.append(total_loss)

            best_accuracy = max(accuracy_scores.keys())

            last_model = self.state_dict()

        best_model = accuracy_scores[best_accuracy]
        print("\n\nBest validation Audio-Video model accuracy: ", best_accuracy, "\n")
        torch.save(best_model, '/kaggle/working/models/audio_video_model.pth')
        accuracy_scores = list(accuracy_scores.keys())

        torch.save(last_model, '/kaggle/working/models/last_epoch_audio_video_model.pth')

        plt.plot(losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Audio-video training Loss')
        plt.show()

        plt.plot(accuracy_scores)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Audio-video validation Accuracy')
        plt.show()

        confusion_matrix_ = confusion_matrix_ / np.sum(confusion_matrix_, axis=1, keepdims=True)

        plt.figure(figsize=(10, 8))
        sns.heatmap(confusion_matrix_, annot=True, fmt=".2%", xticklabels=labels_list, yticklabels=labels_list, cmap = "Blues")
        plt.title('Video-Audio prediction on test set in training mode')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
        return

    def test_model(self, test_dataloader):
        labels_list = ['Happy', 'Sad', 'Angry', 'Fearful', 'Disgusted', 'Surprised']
        num_labels = len(labels_list)
        confusion_matrix_ = np.zeros((num_labels, num_labels))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        model_path = '/kaggle/working/models/audio_video_model.pth'
        self.load_state_dict(torch.load(model_path))
        self.eval()

        all_preds = []
        all_labels = []
        all_outputs = []

        for inputs, labels in test_dataloader:
            pred = inputs.to(device)
            labels = labels.to(device)

            outputs = self(pred)
            all_outputs.append(outputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            confusion_matrix_ += confusion_matrix(labels.cpu().numpy(), preds.cpu().numpy(), labels=range(num_labels))

        accuracy = accuracy_score(all_labels, all_preds)
        print(f'\n\nAudio-Video model accuracy on test set: {accuracy}')

        confusion_matrix_ = confusion_matrix_ / np.sum(confusion_matrix_, axis=1, keepdims=True)

        plt.figure(figsize=(10, 8))
        sns.heatmap(confusion_matrix_, annot=True, fmt=".2%", xticklabels=labels_list, yticklabels=labels_list, cmap="Blues")
        plt.title('Video-Audio prediction on test set')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
        return all_preds, all_outputs
    
    
if __name__ == "__main__":
    
    path_RAVDESS = load_dataset('/kaggle/input/ravdess-emotional-speech-video/RAVDESS dataset')
    path_RYERSON = load_dataset('/kaggle/input/ryerson-emotion-database', True)
    path = path_RAVDESS + path_RYERSON
    
    dir_RAVDESS = ["RAVDESS"]*len(path_RAVDESS)
    dir_RYERSON = ["RYERSON"]*len(path_RYERSON)
    dir_name = dir_RAVDESS + dir_RYERSON
    
    train_data, test_data, train_dir, test_dir = train_test_split(path, dir_name, test_size=0.1, random_state= 15)
    
    print("Submodels train + Audio-video model train size: ", len(train_data))
    print("Audio-video model test size: ", len(test_data))
    print()
    
    subtrain_data, subtest_data, subtrain_dir, subtest_dir = train_test_split(train_data, train_dir, test_size=0.3, random_state= 15)
    
    print("Submodels train size: ", len(subtrain_data))
    print("Submodels test size: ", len(subtest_data))
    
    emotions_to_idx = {'ha': 0, 'sa': 1, 'an': 2, 'fe': 3, 'di': 4, 'su': 5}
    
    video_model_path = '/kaggle/working/models/video_best_model.keras'
    audio_model_path = '/kaggle/working/models/audio_model.pth'
    
    train_video_model(subtrain_data, subtrain_dir, subtest_data, subtest_dir, emotions_to_idx)
    subtest_dataloader_audio, subtest_labels = create_audio_model()
    
    subtest_video_predict_probs, subtest_video_predict  = video_predict(subtest_data, subtest_dir, '/kaggle/working/Video/subtest_features_rav.pkl', emotions_to_idx, "Video prediction on subtest set")
    test_video_predict_probs, test_video_predict = video_predict(test_data, test_dir, '/kaggle/working/Video/test_features_rav.pkl', emotions_to_idx, "Video prediction on test set")
    
    subtest_audio_predict, test_audio_predict , test_labels, subtest_audio_predict_probs, test_audio_predict_probs = audio_predict(subtest_dataloader_audio)
    
    subtest_video_predict_numpy = np.array(subtest_video_predict_probs)
    subtest_audio_predict_numpy = np.array(subtest_audio_predict_probs)
    subtest_audioVideo_predict = np.column_stack((subtest_audio_predict_numpy, subtest_video_predict_numpy))
    
    test_video_predict_numpy = np.array(test_video_predict_probs)
    test_audio_predict_numpy = np.array(test_audio_predict_probs)
    test_audioVideo_predict = np.column_stack((test_audio_predict_numpy, test_video_predict_numpy))
    
    subtest_dataset = VideoAudioDataset(subtest_audioVideo_predict, subtest_labels)
    
    test_dataset = VideoAudioDataset(test_audioVideo_predict, test_labels)
    
    BATCH_SIZE = 32
    
    subtest_dataloader = DataLoader(subtest_dataset, batch_size=BATCH_SIZE, shuffle = False)
    
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle = False)
    
    audioVideo_model = VideoAudioModel()
    audioVideo_model.train_model(subtest_dataloader, test_dataloader)
    
    def_preds, def_outputs = audioVideo_model.test_model(test_dataloader)
    
    true_labels = np.array(test_labels)
    predictions_model_1 = np.array(test_audio_predict)
    predictions_model_2 = np.array(test_video_predict)
    predictions_model_3 = np.array(def_preds)
    
    correctness = np.zeros((len(test_labels), 3))
    
    correctness[:, 0] = (predictions_model_1 == true_labels)
    correctness[:, 1] = (predictions_model_3 == true_labels)
    correctness[:, 2] = (predictions_model_2 == true_labels)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(correctness, annot=False, cmap='Greens', cbar=False, xticklabels=['Audio', 'Audio-Video','Video'], yticklabels=[])
    plt.xlabel('Model')
    plt.ylabel('Predictions')
    plt.title('Correctness of Predictions by Models')
    plt.show()
