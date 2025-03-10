
from src.video.dataset import *
from src.video.frames_extractor import load_featured_video
from src.video.features_extractor import feature_from_Stream
import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
from src.video.model import VideoEmotionRecognitionModel

def train_video_model(train_path, train_dir_name, test_path, test_dir_name, emotions_to_idx):
    train_dataset_filepath = '../../backup/Video/subtrain_features_rav.pkl'
    test_dataset_filepath = '../../backup/Video/subtest_features_rav.pkl'
    os.makedirs('../../backup/Video/', exist_ok = True)

    if os.path.exists(train_dataset_filepath):
        train_point_list, train_labels = load_dataset_jlb(train_dataset_filepath)
        test_point_list, test_labels = load_dataset_jlb(test_dataset_filepath)
    else:
        train_point_list, train_labels = load_featured_video(train_path, train_dir_name, emotions_to_idx)
        save_dataset(train_point_list, train_labels, train_dataset_filepath)

        test_point_list, test_labels = load_featured_video(test_path, test_dir_name, emotions_to_idx)
        save_dataset(test_point_list, test_labels, test_dataset_filepath)

    NUM_CLASSES = len(emotions_to_idx.keys())

    os.makedirs('../../backup/models/', exist_ok = True)

    emotion_model = VideoEmotionRecognitionModel(input_shape=(50, 52), num_classes=NUM_CLASSES)

    emotion_model.summary()

    history = emotion_model.train(train_point_list, train_labels, test_point_list, test_labels, epochs=70, batch_size=32)
    
def video_predict(test_path, test_dir_name, features_file_name, emotions_to_idx, cm_title):

    os.makedirs('../../backup/Video/', exist_ok = True)
    labels_name = ['Happy', 'Sad', 'Angry', 'Fearful', 'Disgusted', 'Surprised']

    model_path = '../../backup/models/video_best_model.keras'

    if os.path.exists(features_file_name):
        test_point_list, test_labels = load_dataset_jlb(features_file_name)
    else:
        test_point_list, test_labels = load_featured_video(test_path, test_dir_name, emotions_to_idx)
        save_dataset(test_point_list, test_labels, features_file_name)

    NUM_CLASSES = len(emotions_to_idx.keys())

    model = VideoEmotionRecognitionModel(input_shape=(50, 52), num_classes=NUM_CLASSES)
    model.model.load_weights(model_path)

    model.summary()

    test_predict_probs, test_predict = model.predict_prob(test_point_list)

    model.plot_confusion_matrix(test_labels, test_predict, labels_name, cm_title)

    return test_predict_probs, test_predict



"""Main function"""
def main():
    dataset_filepath = 'dataset_features_rav.pkl'
    label_dict = {"ha": 3, "sa": 4, "an": 5, "fe": 6, "di": 7, "su": 8}
    labels_name = label_dict.values()
    
    if os.path.exists(dataset_filepath):
        point_list, labels = load_dataset_jlb(dataset_filepath)
    else:
        # FOR KAGGLE, COMMENT IF YOU ARE IN THE REPOSITORY
        # path_RAVDESS = load_dataset('/kaggle/input/ravdess-emotional-speech-video/RAVDESS dataset')
        # path_RYERSON = load_dataset('/kaggle/input/ryerson-emotion-database', True)

        # FOR REPOSITORY, COMMENT IF YOU ARE ON KAGGLE
        path_RAVDESS = load_dataset('datasets/RAVDESS-emotion-database')
        path_RYERSON = load_dataset('datasets/ryerson-emotion-database', True)

        frames_RAVDESS, labels_RAVDESS = load_video(path_RAVDESS, None)
        frames_RYERSON, labels_RYERSON = load_video(path_RYERSON, label_dict)
        
        frames_list = np.concatenate((frames_RAVDESS, frames_RYERSON), axis=0)
        labels = np.concatenate((labels_RAVDESS, labels_RYERSON), axis=0)
        
        point_list = feature_from_Stream(frames_list)
        save_dataset(point_list, labels, dataset_filepath)


    NUM_CLASSES = len(labels_name)
    
    print(len(point_list), NUM_CLASSES) # (720 sample, 50 frame per sample, 52 features del volto per frame)
    # Step 1: Dividi i dati in train + validation e test set
    X_train_val, X_test, y_train_val, y_test = train_test_split(point_list, labels, test_size=0.20, random_state=42)

    # Step 2: Dividi ulteriormente il train + validation set in train e validation set
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)

    # Verifica delle dimensioni dei set
    print("Dimensioni del training set:", X_train.shape, y_train.shape)
    print("Dimensioni del validation set:", X_val.shape, y_val.shape)
    print("Dimensioni del test set:", X_test.shape, y_test.shape)

    # Creazione dell'istanza del modello
    emotion_model = models.VideoEmotionRecognitionModel(input_shape=(50, 52), num_classes=NUM_CLASSES)

    # Sommario del modello
    emotion_model.summary()

    # Addestramento del modello
    history = emotion_model.train(X_train, y_train, X_test, y_test, epochs=50, batch_size=32)

    # Valutazione del modello sui dati di test
    test_loss, test_accuracy = emotion_model.evaluate(X_test, y_test)
    print(f'Test Loss: {test_loss}')
    print(f'Test Accuracy: {test_accuracy}')

    # Predizione su nuovi dati
    predicted_classes = emotion_model.predict(X_test)

    # Visualizzare alcune predizioni
    # print(predicted_classes[:10])
    # print(y_test[:10])

    # Visualizzare la confusion matrix
    emotion_model.plot_confusion_matrix(y_test, predicted_classes, labels_name)

    
    
if __name__ == "__main__":
    main()