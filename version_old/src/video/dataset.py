import os
from kaggle.api.kaggle_api_extended import KaggleApi
import joblib

def download_dataset():
    """
    Download the dataset "shanp1234/ryerson-emotion-database" from kaggle and put in "../datasets/ryerson-emotion-database"
    """
    # Imposta il nome del dataset che vuoi scaricare
    dataset = 'adrivg/ravdess-emotional-speech-video'


    # Crea un'istanza dell'API di Kaggle
    api = KaggleApi()
    api.authenticate()

    # Crea una directory per il dataset se non esiste
    dataset_dir = '../../datasets/RAVDESS-emotion-database'
    os.makedirs(dataset_dir, exist_ok=True)

    # Verifica se la directory è vuota
    if not os.listdir(dataset_dir):
        # Crea un'istanza dell'API di Kaggle
        api = KaggleApi()
        api.authenticate()

        # Scarica il dataset
        api.dataset_download_files(dataset, path=dataset_dir, unzip=True)

        print(f'Dataset {dataset} scaricato e decompresso in {dataset_dir}')
    else:
        print(f'Il dataset {dataset} è già presente nella directory {dataset_dir}.')

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



if __name__ == "__main__":
    print(load_dataset())