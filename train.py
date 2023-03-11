import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
from config import *

onlyfiles = [f for f in listdir(PATH) if isfile(join(PATH, f))]

Training_Data, Labels = [], []

for i, files in enumerate(onlyfiles):
    image_path = PATH + onlyfiles[i]  # face/user1.jpg
    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    Training_Data.append(np.asarray(images, dtype=np.uint8))
    Labels.append(i)

Labels = np.asarray(Labels, dtype=np.int32)

# Linear Binary Phase Histogram Classifier
model = cv2.face.LBPHFaceRecognizer_create()

# this line will generate error run the following command
# python -m pip install --user opencv-contrib-python


model.train(np.asarray(Training_Data), np.asarray(Labels))
print('[INFO] Model Training Complete !!!')
