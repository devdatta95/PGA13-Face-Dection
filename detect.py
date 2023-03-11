from config import *
import cv2
from utils import face_detector
from train import model

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    image, face = face_detector(frame)

    try:

        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        result = model.predict(face)

        if result[1] < 500:
            confidence = int(100 * (1 - (result[1]) / 300))
            display_string = str(confidence) + '% Confidence it is USER'

        cv2.putText(image, display_string, (100, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 120, 255), 2)

        if confidence > 75:
            cv2.putText(image, "Unlocked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 55, 255), 2)
            cv2.imshow('Face Cropper', image)

        else:
            cv2.putText(image, "Locked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 55, 0), 2)
            cv2.imshow('Face Cropper', image)

    except:
        cv2.putText(image, "Face Not Found", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 2)

        cv2.imshow('Face Cropper', image)
        pass

    if cv2.waitKey(1)==13:
        break

cap.release()
cv2.destroyAllWindows()