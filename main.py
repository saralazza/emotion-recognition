import numpy as np

from src.video.dataset import load_dataset
from sklearn.model_selection import train_test_split

from src.video.model import *
from src.audio.model import *
from src.final_combination.model import *


def main():
    path_RAVDESS = load_dataset('../../datasets/RAVDESS dataset')
    path_RYERSON = load_dataset('../../datasets/ryerson-emotion-database', True)
    path = path_RAVDESS + path_RYERSON

    dir_RAVDESS = ["RAVDESS"]*len(path_RAVDESS)
    dir_RYERSON = ["RYERSON"]*len(path_RYERSON)
    dir_name = dir_RAVDESS + dir_RYERSON

    train_data, test_data, train_dir, test_dir = train_test_split(path, dir_name, test_size=0.1, random_state= 15)

    print("Submodels train + Audio-video model train size: ", len(train_data))
    print("Audio-video model test size: ", len(test_data))
    print()

    subtrain_data, subtest_data, subtrain_dir, subtest_dir = train_test_split(train_data, train_dir, test_size=0.3, random_state= 15)

    print("Submodels train size: ", len(subtrain_data))
    print("Submodels test size: ", len(subtest_data))

    emotions_to_idx = {'ha': 0, 'sa': 1, 'an': 2, 'fe': 3, 'di': 4, 'su': 5}

    video_model_path = '../../backup/models/video_best_model.keras'
    audio_model_path = '../../backup/models/audio_model.pth'

    train_video_model(subtrain_data, subtrain_dir, subtest_data, subtest_dir, emotions_to_idx)
    subtest_dataloader_audio, subtest_labels = create_audio_model(subtrain_data, subtrain_dir, subtest_data, subtest_dir)

    subtest_video_predict_probs, subtest_video_predict  = video_predict(subtest_data, subtest_dir, '../../backup/Video/subtest_features_rav.pkl', emotions_to_idx, "Video prediction on subtest set")
    test_video_predict_probs, test_video_predict = video_predict(test_data, test_dir, '../../backup/Video/test_features_rav.pkl', emotions_to_idx, "Video prediction on test set")

    subtest_audio_predict, test_audio_predict , test_labels, subtest_audio_predict_probs, test_audio_predict_probs = audio_predict(subtest_dataloader_audio)

    subtest_video_predict_numpy = np.array(subtest_video_predict_probs)
    subtest_audio_predict_numpy = np.array(subtest_audio_predict_probs)
    subtest_audioVideo_predict = np.column_stack((subtest_audio_predict_numpy, subtest_video_predict_numpy))

    test_video_predict_numpy = np.array(test_video_predict_probs)
    test_audio_predict_numpy = np.array(test_audio_predict_probs)
    test_audioVideo_predict = np.column_stack((test_audio_predict_numpy, test_video_predict_numpy))

    subtest_dataset = VideoAudioDataset(subtest_audioVideo_predict, subtest_labels)

    test_dataset = VideoAudioDataset(test_audioVideo_predict, test_labels)

    BATCH_SIZE = 32

    subtest_dataloader = DataLoader(subtest_dataset, batch_size=BATCH_SIZE, shuffle = False)

    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle = False)

    audioVideo_model = VideoAudioModel()
    audioVideo_model.train_model(subtest_dataloader, test_dataloader)

    def_preds, def_outputs = audioVideo_model.test_model(test_dataloader)

    true_labels = np.array(test_labels)
    predictions_model_1 = np.array(test_audio_predict)
    predictions_model_2 = np.array(test_video_predict)
    predictions_model_3 = np.array(def_preds)

    correctness = np.zeros((len(test_labels), 3))

    correctness[:, 0] = (predictions_model_1 == true_labels)
    correctness[:, 1] = (predictions_model_3 == true_labels)
    correctness[:, 2] = (predictions_model_2 == true_labels)

    plt.figure(figsize=(12, 10))
    sns.heatmap(correctness, annot=False, cmap='Greens', cbar=False, xticklabels=['Audio', 'Audio-Video','Video'], yticklabels=[])
    plt.xlabel('Model')
    plt.ylabel('Predictions')
    plt.title('Correctness of Predictions by Models')
    plt.show()

if __name__ == "__main__":
    main()