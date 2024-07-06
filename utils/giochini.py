import cv2
from src.video.features_extractor import *
from src.video.frames_extractor import *

stream = frames_extraction("datasets/RAVDESS-emotion-database/Video_Speech_Actor_01/Actor_01/01-01-03-01-02-02-01.mp4")
cv2.imwrite("out/contento.png",annotated_From_image(stream[49]))
cv2.waitKey(0)