import cv2
import dlib
import numpy as np

image = cv2.imread('1.jpeg')

predictor_path = 'shape_predictor_81_face_landmarks.dat'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
dets = detector(image, 0)

for k, d in enumerate(dets):
    shape = predictor(image, d)
    landmarks = np.matrix([[p.x, p.y] for p in shape.parts()])
    for num in range(shape.num_parts):
        cv2.circle(image, (shape.parts()[num].x, shape.parts()[num].y), 3, (0,255,0), -1)
cv2.imshow('frame', image)
cv2.waitKey(0)
