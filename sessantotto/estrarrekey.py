import cv2
import dlib
import numpy as np
from tqdm import tqdm

def get_faces_keypoints(frames_list):
    
    # Inizializza il rilevatore di facce e il predittore di punti facciali
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    points_list = []
    for samp in tqdm(frames_list,desc="finding points"):
        nuovoSamp = []
        for frame in samp:
            # Rileva facce nell'immagine
            #print(type(frame[0][0]))
            faces = detector(frame)
    
            keypoints = []

            for face in faces:
                # Predici i punti facciali
                landmarks = predictor(frame, face)
                for n in range(0, 68):  # Il modello prevede 68 punti facciali
                    x = landmarks.part(n).x
                    y = landmarks.part(n).y
                    keypoints.append((x, y))
                    # Disegna i punti sull'immagine
                    #cv2.circle(image, (x, y), 2, (255, 0, 0), -1)
            nuovoSamp.append(keypoints)
        points_list.append(nuovoSamp)
    
    return np.array(points_list, dtype='intc')


def get_face_keypoints(image_path):
    # Carica l'immagine
    image = cv2.imread(image_path)
    
    if image is None:
        raise FileNotFoundError(f"Impossibile aprire o leggere il file: {image_path}")
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Inizializza il rilevatore di facce e il predittore di punti facciali
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    # Rileva facce nell'immagine
    faces = detector(gray)
    
    keypoints = []

    for face in faces:
        # Predici i punti facciali
        landmarks = predictor(gray, face)
        for n in range(0, 61):  # Il modello prevede 68 punti facciali
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            keypoints.append((x, y))
            # Disegna i punti sull'immagine
            cv2.circle(image, (x, y), 2, (255, 0, 0), -1)

    return keypoints, image


def main():
    image_pathN = 'faces/Nface1.png'  # Sostituisci con il percorso della tua immagine
    image_pathH = 'faces/Hface1.png'  # Sostituisci con il percorso della tua immagine
    image_pathA = 'faces/Aface1.png'  # Sostituisci con il percorso della tua immagine
    
    
    keypointsN, image_with_keypointsN = get_face_keypoints(image_pathN)
    keypointsH, image_with_keypointsH = get_face_keypoints(image_pathH)
    keypointsA, image_with_keypointsA = get_face_keypoints(image_pathA)
    
     
    #print("Keypoints estratti:", keypoints)
    
    # Mostra l'immagine con i punti chiave disegnati
    cv2.imshow("Keypoints", image_with_keypointsN)
    cv2.waitKey(0)
    cv2.imshow("Keypoints", image_with_keypointsH)
    cv2.waitKey(0)
    cv2.imshow("Keypoints", image_with_keypointsA)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
