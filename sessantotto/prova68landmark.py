from video_features import feature_extraction_NN 
import cv2
import dlib
import numpy as np
frames = feature_extraction_NN("./datasets/ryerson-emotion-database/s1/s1/sa1.avi")
frames2 = feature_extraction_NN("./datasets/ryerson-emotion-database/s1/s1/ha1.avi")


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
for i in range(50):
# Rileva facce nell'immagine
    faces = detector(frames[i])
    black = np.zeros(shape=(240,320))
    black2 = np.zeros(shape=(240,320))
    
    keypoints = []
    keypoints2 = []
    print(type(frames[i][0][0]))
    for face in faces:
                # Predici i punti facciali
                landmarks = predictor(frames[i], face)
                landmarks2 = predictor(frames2[i], face)
                
                
                for n in range(0, 68):  # Il modello prevede 68 punti facciali
                    x = landmarks.part(n).x
                    y = landmarks.part(n).y
                    keypoints.append((x, y))
                    
                    x2 = landmarks2.part(n).x
                    y2 = landmarks2.part(n).y
                    keypoints2.append((x2, y2))


                    # Disegna i punti sull'immagine
                    cv2.circle(frames[i], (x, y), 2, (255, 0, 0), -1)
                    cv2.circle(frames2[i], (x2, y2), 2, (255, 0, 0), -1)
                    


    #print(keypoints)
    
    combined_image = np.hstack((frames[i], frames2[i]))
    cv2.imshow(f"frame {i}", cv2.resize(combined_image, (combined_image.shape[1]*2, combined_image.shape[0]*2)))
    # cv2.imshow(f"frame {i}", cv2.resize(black2, (640, 480)))
    cv2.waitKey(0)

cv2.destroyAllWindows()
