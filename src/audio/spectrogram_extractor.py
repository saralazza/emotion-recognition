import os
from src.audio.loader import * 
import librosa
import numpy as np
import torch
from torch.utils.data import Dataset

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