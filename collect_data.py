from config import *
import cv2
from utils import face_extractor


cap = cv2.VideoCapture(DEVICE)
count = 0

while True:
    ret, frame = cap.read()

    cv2.imshow('img ', frame)
    if face_extractor(frame) is not None:
        count += 1
        face = cv2.resize(face_extractor(frame), (HEIGHT, WIDTH))

        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        file_name_path = PATH + "user" + str(count) + '.jpg'

        cv2.imwrite(file_name_path, face)
        cv2.putText(face,
                    str(count),
                    (50, 50),
                    cv2.FONT_HERSHEY_COMPLEX,
                    1,
                    (0, 255, 0),
                    2)

        cv2.imshow('FACE CROPPER ', face)
    else:
        print('face not found ')

    if cv2.waitKey(1) == 13 or count == TOTAL_IMAGES:
        break

cap.release()
cv2.destroyAllWindows()
print('[INFO] Collecting Samples Complete  !!!!')
