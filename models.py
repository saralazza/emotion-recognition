import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

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

    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val))
        return history

    def evaluate(self, X_test, y_test):
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test)
        return test_loss, test_accuracy

    def predict(self, X):
        predictions = self.model.predict(X)
        predicted_classes = np.argmax(predictions, axis=1)
        return predicted_classes
    
    def plot_confusion_matrix(self, y_true, y_pred, class_names):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='g', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.title('Confusion Matrix')
        plt.show()

        
  

  

