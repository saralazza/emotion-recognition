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
    download_dataset()

    directory = './datasets/ryerson-emotion-database'

    for nome_cartella in os.listdir(directory):
        # Ottieni il percorso completo della cartella
        percorso_cartella = os.path.join(directory, nome_cartella, nome_cartella)

        if os.path.isdir(percorso_cartella):
            for nome_file in os.listdir(percorso_cartella):
                percorso_file = os.path.join(percorso_cartella, nome_file)
                if not os.path.isdir(percorso_file):
                    if not percorso_file.endswith(".db"):
                        print(percorso_file)
                else: 
                    for nome_file2 in os.listdir(percorso_file):
                        percorso_file2 = os.path.join(percorso_file, nome_file2)
                        print(percorso_file2)