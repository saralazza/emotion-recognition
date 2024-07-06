import torch.nn as nn
import os
import numpy as np
import torch
import tqdm
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from spectrogram_extractor import * 
from loader import *
import seaborn as sns
from torch.utils.data import DataLoader

class AudioModel(nn.Module):
    def __init__(self, num_labels=6):
        super(AudioModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding="same")
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding="same")
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding="same")
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding="same")

        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(128 * 4 * 5, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(0.1)
        self.fc4 = nn.Linear(128, num_labels)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.unsqueeze(1)

        x = self.relu(self.conv1(x))
        x = self.pool(x)

        x = self.relu(self.conv2(x))
        x = self.pool(x)

        x = self.relu(self.conv3(x))
        x = self.pool(x)

        x = self.relu(self.conv4(x))
        x = self.pool(x)

        x = x.flatten(start_dim=1)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc4(x)
        x = self.softmax(x)

        return x


def train_audio_model(subtrain_dataloader, subtest_dataloader):

    model = AudioModel()

    # Labels names for the plots
    labels_list = ['Happy', 'Sad', 'Angry', 'Fearful', 'Disgusted', 'Surprised']

    num_labels = len(labels_list)
    confusion_matrix_ = np.zeros((num_labels, num_labels))

    os.makedirs("../../backup/models", exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.0001, weight_decay=0.0001)

    losses = []
    accuracy_scores = dict()

    num_epochs = 70

    for epoch in range(num_epochs):
        model.train()

        total_loss = 0.0

        for audios, labels in tqdm(subtrain_dataloader):
            audios = audios.to(device)
            labels = labels.to(device)

            outputs = model(audios)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss}')

        model.eval()

        all_preds = []
        all_labels = []

        for audios, labels in subtest_dataloader:
            audios = audios.to(device)
            labels = labels.to(device)

            outputs = model(audios)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            confusion_matrix_ += confusion_matrix(labels.cpu().numpy(), preds.cpu().numpy(), labels=range(num_labels))

        accuracy = accuracy_score(all_labels, all_preds)
        print(f'Validation accuracy: {accuracy}')

        accuracy_scores[accuracy] = model.state_dict()

        losses.append(total_loss)

    best_accuracy = max(accuracy_scores.keys())
    best_model = accuracy_scores[best_accuracy]
    print("\nBest validation accuracy: ", best_accuracy, "\n")
    torch.save(best_model, '../../backup/models/audio_model.pth')
    accuracy_scores = list(accuracy_scores.keys())

    # Plot Training Loss
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Audio training Loss')
    plt.show()

    # Plot Validation Accuracy
    plt.plot(accuracy_scores)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Audio validation Accuracy')
    plt.show()

    confusion_matrix_ = confusion_matrix_ / np.sum(confusion_matrix_, axis=1, keepdims=True)

    # confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix_, annot=True, fmt=".2%", xticklabels=labels_list, yticklabels=labels_list, cmap='Purples')
    plt.title('Audio prediction on subtest set on training mode')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()


def create_audio_model(subtrain_data, subtrain_dir, subtest_data, subtest_dir):

    print("Audio extraction by video of subtrain set...\n")
    audio_paths_subtrain = extract_audio_from_videos(subtrain_data, subtrain_dir, '../../backup/Audios/subtrain')
    print("Audio extraction by video of subtest set...\n")
    audio_paths_subtest = extract_audio_from_videos(subtest_data, subtest_dir, '../../backup/Audios/subtest')

    print("Features extraction by audio of subtrain set...\n")
    output_folder = "../../datasets/subtrain"
    features_paths_subtrain = preprocess_dataset(audio_paths_subtrain, output_folder)

    print("Features extraction by audio of subtest set...\n")
    output_folder = "../../datasets/subtest"
    features_paths_subtest = preprocess_dataset(audio_paths_subtest, output_folder)

    # estrazione delle label delle features
    subtrain_labels = load_audio_dataset(features_paths_subtrain)
    subtest_labels = load_audio_dataset(features_paths_subtest)

    BATCH_SIZE = 32

    subtrain_dataset = MultilabelSpectrogramDataset(features_paths_subtrain, subtrain_labels)

    subtest_dataset = MultilabelSpectrogramDataset(features_paths_subtest, subtest_labels)

    subtrain_dataloader = DataLoader(subtrain_dataset, batch_size=BATCH_SIZE, shuffle=True)
    subtest_dataloader = DataLoader(subtest_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print("Training audio model...\n")
    train_audio_model(subtrain_dataloader, subtest_dataloader)

    return subtest_dataloader, subtest_labels


def audio_predict(subtest_dataloader_audio, test_data, test_dir):
    # audio, fetature and label extraction by videos

    print("Audio extraction by video of test set...\n")
    audio_paths_test = extract_audio_from_videos(test_data, test_dir, '../../backup/Audios/test')

    print("Features extraction by audio of test set...\n")
    output_folder = "../../datasets/test"
    features_paths_test = preprocess_dataset(audio_paths_test, output_folder)

    # label extraction by test features
    test_labels = load_audio_dataset(features_paths_test)

    BATCH_SIZE = 32

    #  dataset creation with [spectogram, label]
    test_dataset = MultilabelSpectrogramDataset(features_paths_test, test_labels)

    # converting dataset in dataloader
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print("Loading of audio model by file...\n")
    model_path = '../../backup/models/audio_model.pth'

    model = AudioModel()

    model.load_state_dict(torch.load(model_path))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    model.eval()

    labels_list = ['Happy', 'Sad', 'Angry', 'Fearful', 'Disgusted', 'Surprised']
    num_labels = len(labels_list)
    confusion_matrix_ = np.zeros((num_labels, num_labels))
    subtest_preds = []
    all_labels = []
    all_outputs_subtest = []

    for audios, labels in subtest_dataloader_audio:
        audios = audios.to(device)
        labels = labels.to(device)

        outputs = model(audios)
        all_outputs_subtest.extend(outputs.cpu().detach().numpy())
        _, preds = torch.max(outputs, 1)

        subtest_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        confusion_matrix_ += confusion_matrix(labels.cpu().numpy(), preds.cpu().numpy(), labels=range(num_labels))

    accuracy = accuracy_score(all_labels, subtest_preds)
    print(f'\n\nAccuracy of audio model on subtest set: {accuracy}\n')

    confusion_matrix_ = confusion_matrix_ / np.sum(confusion_matrix_, axis=1, keepdims=True)

    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix_, annot=True, fmt=".2%", xticklabels=labels_list, yticklabels=labels_list, cmap = "Purples")
    plt.title('Audio prediction on subtest set')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

    confusion_matrix_ = np.zeros((num_labels, num_labels))
    test_preds = []
    all_labels = []
    all_outputs_test = []
    for audios, labels in test_dataloader:
        audios = audios.to(device)
        labels = labels.to(device)

        outputs = model(audios)
        _, preds = torch.max(outputs, 1)

        all_outputs_test.extend(outputs.cpu().detach().numpy())
        test_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        confusion_matrix_ += confusion_matrix(labels.cpu().numpy(), preds.cpu().numpy(), labels=range(num_labels))

    accuracy = accuracy_score(all_labels, test_preds)
    print(f'\n\nAccuracy of audio model on test set: {accuracy}\n')

    confusion_matrix_ = confusion_matrix_ / np.sum(confusion_matrix_, axis=1, keepdims=True)

    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix_, annot=True, fmt=".2%", xticklabels=labels_list, yticklabels=labels_list, cmap='Purples')
    plt.title('Audio prediction on test set')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

    return subtest_preds, test_preds, test_labels, all_outputs_subtest, all_outputs_test
