import os
import cv2
import numpy
from multiprocessing import Pool
import time

size_input_img = 64
size_output_img = 256
dir_sep = "/"

dir_all_images = "/Users/Macbook/Desktop/ARA_PROJE/Codes/VEHICLE/dataset"

dir_input_images = dir_all_images+dir_sep+str(size_input_img)
dir_output_images = dir_all_images+dir_sep+str(size_output_img)

try:
    os.mkdir(dir_input_images)
    os.mkdir(dir_output_images)
except:
    pass

filenames = os.listdir(dir_all_images)

min_size = 300

# for fname in filenames:
def dataset_seperator(fname):
    try:
        img = cv2.imread(dir_all_images+dir_sep+fname)
        height, width, channel = img.shape
        if height >= min_size and width >= min_size and channel == 3:
            print(img.shape,fname)
            im1 = cv2.resize(img,(size_input_img,size_input_img))
            im2 = cv2.resize(img,(size_output_img,size_output_img))
            cv2.imwrite(dir_input_images+dir_sep+fname,im1)
            cv2.imwrite(dir_output_images+dir_sep+fname,im2)
    except:
        print("Error occured while reading image file...")
        pass

if __name__ == '__main__':
    start_time = time.time()
    pool = Pool(8)
    pool.map(dataset_seperator,filenames)
    pool.close()
    pool.join()
    print("--- %s seconds ---" % (time.time() - start_time))
