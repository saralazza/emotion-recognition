import os
from kaggle.api.kaggle_api_extended import KaggleApi

def download_dataset():
    # Imposta il nome del dataset che vuoi scaricare
    dataset = 'shanp1234/ryerson-emotion-database'

    # Crea un'istanza dell'API di Kaggle
    api = KaggleApi()
    api.authenticate()

    # Crea una directory per il dataset se non esiste
    dataset_dir = './datasets/ryerson-emotion-database'
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

def load_dataset():
    pass