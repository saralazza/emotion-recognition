from video_features import feature_extraction_NN 
from visualizza import visualizza
from puntiStream import feature_From_Stream
import mediapipe as mp
import cv2
import numpy as np
frames = feature_extraction_NN("./datasets/ryerson-emotion-database/s1/s1/sa1.avi")
features_score_list =[]
for i in range(50):
# Rileva facce nell'immagine

    

# Crea un'immagine MediaPipe dalla matrice NumPy
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frames[i])
    visualizza(feature_From_Stream(mp_image))

                    


    #print(keypoints)
    
    


