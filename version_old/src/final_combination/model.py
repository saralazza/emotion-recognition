import torch
from torch.utils.data import Dataset
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
import tqdm


class VideoAudioDataset(Dataset):
    def __init__(self, preds, labels):
        self.preds = preds
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        pred = self.preds[idx]
        label = self.labels[idx]
        return torch.tensor(pred, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


class VideoAudioModel(nn.Module):
    def __init__(self, num_labels=6):
        super(VideoAudioModel, self).__init__()
        self.fc1 = nn.Linear(12,128)
        self.fc2 = nn.Linear(128,256)
        self.fc3 = nn.Linear(256,512)
        self.fc4 = nn.Linear(512,128)
        self.fc5 = nn.Linear(128,64)
        self.fc6 = nn.Linear(64,num_labels)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.dropout(torch.relu(self.fc2(x)))
        x = self.dropout(torch.relu(self.fc3(x)))
        x = self.dropout(torch.relu(self.fc4(x)))
        x = self.dropout(torch.relu(self.fc5(x)))
        x = self.fc6(x)
        return x

    def train_model(self, train_dataloader, test_dataloader):
        labels_list = ['Happy', 'Sad', 'Angry', 'Fearful', 'Disgusted', 'Surprised']
        num_labels = len(labels_list)
        confusion_matrix_ = np.zeros((num_labels, num_labels))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        losses = []
        accuracy_scores = dict()

        num_epochs = 100

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            self.parameters(), lr=0.0001, weight_decay=0.0001)

        last_model = self.state_dict()

        for epoch in range(num_epochs):
            self.train()
            total_loss = 0.0
            for inputs, labels in tqdm(train_dataloader):
                pred = inputs.to(device)
                labels = labels.to(device)

                outputs = self(pred)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss}')

            self.eval()

            all_preds = []
            all_labels = []

            for inputs, labels in test_dataloader:
                pred = inputs.to(device)
                labels = labels.to(device)

                outputs = self(pred)
                _, preds = torch.max(outputs, 1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                confusion_matrix_ += confusion_matrix(labels.cpu().numpy(), preds.cpu().numpy(), labels=range(num_labels))

            accuracy = accuracy_score(all_labels, all_preds)
            print(f'Validation accuracy: {accuracy}')

            accuracy_scores[accuracy] = self.state_dict()

            losses.append(total_loss)

            best_accuracy = max(accuracy_scores.keys())

            last_model = self.state_dict()

        best_model = accuracy_scores[best_accuracy]
        print("\n\nBest validation Audio-Video model accuracy: ", best_accuracy, "\n")
        torch.save(best_model, '../../backup/models/audio_video_model.pth')
        accuracy_scores = list(accuracy_scores.keys())

        torch.save(last_model, '../../backup/models/last_epoch_audio_video_model.pth')

        plt.plot(losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Audio-video training Loss')
        plt.show()

        plt.plot(accuracy_scores)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Audio-video validation Accuracy')
        plt.show()

        confusion_matrix_ = confusion_matrix_ / np.sum(confusion_matrix_, axis=1, keepdims=True)

        plt.figure(figsize=(10, 8))
        sns.heatmap(confusion_matrix_, annot=True, fmt=".2%", xticklabels=labels_list, yticklabels=labels_list, cmap = "Blues")
        plt.title('Video-Audio prediction on test set in training mode')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
        return

    def test_model(self, test_dataloader):
        labels_list = ['Happy', 'Sad', 'Angry', 'Fearful', 'Disgusted', 'Surprised']
        num_labels = len(labels_list)
        confusion_matrix_ = np.zeros((num_labels, num_labels))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        model_path = '../../backup/models/audio_video_model.pth'
        self.load_state_dict(torch.load(model_path))
        self.eval()

        all_preds = []
        all_labels = []
        all_outputs = []

        for inputs, labels in test_dataloader:
            pred = inputs.to(device)
            labels = labels.to(device)

            outputs = self(pred)
            all_outputs.append(outputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            confusion_matrix_ += confusion_matrix(labels.cpu().numpy(), preds.cpu().numpy(), labels=range(num_labels))

        accuracy = accuracy_score(all_labels, all_preds)
        print(f'\n\nAudio-Video model accuracy on test set: {accuracy}')

        confusion_matrix_ = confusion_matrix_ / np.sum(confusion_matrix_, axis=1, keepdims=True)

        plt.figure(figsize=(10, 8))
        sns.heatmap(confusion_matrix_, annot=True, fmt=".2%", xticklabels=labels_list, yticklabels=labels_list, cmap="Blues")
        plt.title('Video-Audio prediction on test set')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
        return all_preds, all_outputs
 