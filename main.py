
from dataset import *
from frames_extractor import load_video
from features_extractor import feature_from_Stream
import os
import numpy as np
import matplotlib as plt
from sklearn.model_selection import train_test_split
import models




def main():
    dataset_filepath = 'dataset_features.pkl'

    if os.path.exists(dataset_filepath):
        point_list, labels = load_dataset_jlb(dataset_filepath)
        labels_name = list({lab for lab in labels})
    else:
        paths, labels_name = load_dataset()
        frames_list, labels = load_video(paths, labels_name)
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