import numpy as np
import tensorflow as tf
from src.video.dataset import *
from src.video.frames_extractor import load_featured_video
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os 

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
            filepath='../../backup/models/video_best_model.keras',
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

  

  

