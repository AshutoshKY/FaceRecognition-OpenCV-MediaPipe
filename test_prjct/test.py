import cv2
import numpy
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    cv2.imshow('frame', frame)
    cv2.waitKey(1)