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

    labels_name = []
    paths = []

    for subject in os.listdir(directory):
        percorso_dir_subject = os.path.join(directory, subject, subject)

        for elem in os.listdir(percorso_dir_subject):
            percorso_elem = os.path.join(percorso_dir_subject, elem)
            if not os.path.isdir(percorso_elem):
                if not percorso_elem.endswith(".db"):
                    paths.append(percorso_elem)
                    label_name = elem[:2]
                    if label_name not in labels_name:
                        labels_name.append(label_name)
            else: 
                for file in os.listdir(percorso_elem):
                    percorso_file = os.path.join(percorso_elem, file)
                    if not percorso_file.endswith(".db"):
                        paths.append(percorso_file)
                        label_name = file[:2]
                        if label_name not in labels_name:
                            labels_name.append(label_name)
    return paths, labels_name