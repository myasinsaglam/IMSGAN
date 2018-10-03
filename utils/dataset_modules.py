import os
import cv2
import numpy

input_dataset_dir = "/Users/Macbook/Desktop/ARA_PROJE/Codes/VEHICLE/dataset"
filenames = os.listdir(input_dataset_dir)

for fname in filenames:
    img = cv2.imread(input_dataset_dir+"/"+fname)
    # cv2.imshow("image",img)
    # cv2.waitKey(0)