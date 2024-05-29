
from dataset import load_dataset
from video_features import load_video
from puntiStream import feature_from_Stream
import joblib
import os
import numpy as np
import matplotlib as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy



def save_dataset(features, labels, filepath):
    joblib.dump((features, labels), filepath)
    print(f"Dataset salvato in {filepath}")

def load_dataset_jlb(filepath):
    features, labels = joblib.load(filepath)
    print(f"Dataset caricato da {filepath}")
    return features, labels

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
    X_train, X_test, y_train, y_test=train_test_split(point_list, labels, test_size=0.06, random_state=10)
    print(X_train.shape, X_test.shape, np.array(y_train).shape, np.array(y_test).shape)

    
    
    """
        # Creazione del modello LSTM
    model = Sequential()
    model.add(LSTM(64, input_shape=(50, 52), return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(64))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(NUM_CLASSES, activation='softmax')) 

    # Compilazione del modello
    model.compile(optimizer=Adam(learning_rate=0.001),
                loss=SparseCategoricalCrossentropy(),
                metrics=[SparseCategoricalAccuracy()])

    # Sommario del modello
    model.summary()

    # Addestramento del modello
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

        # Valutazione del modello sui dati di test
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f'Test Loss: {test_loss}')
    print(f'Test Accuracy: {test_accuracy}')
    # Predizione su nuovi dati
    predictions = model.predict(X_test)
    predicted_classes = np.argmax(predictions, axis=1)

    # Visualizzare alcune predizioni
    print(predicted_classes[:10])
    print(y_test[:10])

    """
    """
    
    model4 = Sequential()
    model4.add(Input(shape=(x_train.shape[1],x_train.shape[2], x_train.shape[3], 3)))
    model4.add(TimeDistributed(Flatten()))
    model4.add(Bidirectional(LSTM(128, return_sequences=True)))
    model4.add(Dropout(0.2))
    model4.add(Bidirectional(LSTM(128, return_sequences=True)))
    model4.add(Dropout(0.2))
    model4.add(Bidirectional(LSTM(128)))
    model4.add(Dropout(0.2))
    model4.add(Dense(NUM_CLASSES, activation='softmax'))
    model4.summary()
    model4.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history4 = model4.fit(x_train, to_categorical(y_train), batch_size=31, epochs=20, validation_data=(x_test, to_categorical(y_test)))
    

    # Plot training & validation accuracy values
    plt.plot(history4.history['accuracy'])
    plt.plot(history4.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    print(f"Test Accuracy: {history4.history['val_accuracy'][-1]}")
    
    # Plot training & validation loss values
    plt.plot(history4.history['loss'])
    plt.plot(history4.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    print(f"Test Loss: {history4.history['val_loss'][-1]}")"""
if __name__ == "__main__":
    main()