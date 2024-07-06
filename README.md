# Emotion-recognition
# :smiley:Aim
The project aim is to classify human emotions analyzing videos and audios.
The emotions recognized are:
- Happiness
- Anger
- Sadness
- Fear
- Disgust
- Surprise

The project can have an important impact as emotions are the basis of social interactions
## :bar_chart:Report
È presente una [relazione](report/Emotion_Recognition_by_Audio_and_Video.pdf) che spiega la realizzazione e composizione del progetto illustrandone risultati e le potenzialità 
## 🗂️ In this repository 
The project is composed in three parts:
- Source code
    - Visual emotion recognition 
        - Content extractor
        - Extractor of features from content
        - Model for learning and recognition
    - Auditory emotion recognition
        - Content extractor
        - Extractor of features from content
        - Model for learning and recognition
    - Recognition from the entire video
        - Model for learning and recognition
        - Main function for running the three combined models
- Assets
_some necessary assets for the system_
- Backup 
    - audio backup
    - video backup 
    - models backup
- Utils
_some useful function to see the result of some processed frame or some images to be processed with mediapipe_
## :books:Libs 
The following libraries are used in the project:
- [os](https://docs.python.org/3/library/os.html)
- [numpy](https://numpy.org/)
- [joblib](https://joblib.readthedocs.io/en/stable/)
- [open CV](https://opencv.org/)
- [tqdm](https://tqdm.github.io/)
- [matplot](https://matplotlib.org/)
- [librosa](https://librosa.org/doc/latest/index.html)
- [mediapipe](https://ai.google.dev/edge/mediapipe/solutions/guide?hl=it)
- [tensorflow](https://www.tensorflow.org/?hl=it)
- [scikit-learn](https://scikit-learn.org/stable/)
- [moviepy](https://pypi.org/project/moviepy/)
- [torch](https://pypi.org/project/torch/)
### Requirements
moviepy == 1.0.3 
mediapipe ==0.8.7 

## How to use
After installing all the necessary libraries run main.py. 
You can also use the program from [emotion_recognition_by_audio_and_video.py](merged/emotion_recognition_by_audio_and_video.py) or [notebook Emotion Recognition by Audio and Video](merged/notebook%20Emotion%20Recognition%20by%20Audio%20and%20Video.ipynb)

