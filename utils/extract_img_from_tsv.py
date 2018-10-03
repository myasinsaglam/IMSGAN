import os
import base64
import struct
import cv2
import numpy as np

min_size = 256
counter = 0
counter2 = 0

fid = open("D:\\IDM Downloads\\FaceImageCroppedWithAlignment.tsv", encoding='utf-8')
base_path = 'D:\\IDM Downloads\\OWO'
if not os.path.exists(base_path):
    os.mkdir(base_path)
bbox_file = open(base_path + '\\bboxes.txt', 'w')
while True:
    line = fid.readline()
    if line:
        data_info = line.split('\t')
        # 0: Freebase MID (unique key for each entity)
        # 1: ImageSearchRank
        # 4: FaceID
        # 5: bbox
        # 6: img_data
        filename = data_info[0] + "\\" + data_info[1] + "_" + data_info[4] + ".jpg"
        bbox = struct.unpack('ffff', base64.b64decode(data_info[5]))
        # bbox = struct.unpack('ffff', data_info[5].decode("base64"))
        bbox_file.write(filename + " " + (" ".join(str(bbox_value) for bbox_value in bbox)) + "\n")

        img_data = base64.b64decode(data_info[6])

        output_file_path = base_path + "\\" + filename
        if os.path.exists(output_file_path):
            print(output_file_path + " exists")

        output_path = os.path.dirname(output_file_path)
        if not os.path.exists(output_path):
            os.mkdir(output_path)

        img_file = open(output_file_path, 'wb')
        img_file.write(img_data)
        img_file.close()

        image = cv2.imread(output_file_path)

        height, width, channel = image.shape

        if height >= min_size and width >= min_size and channel == 3:
            counter += 1
        else:
            os.remove(output_file_path)

        counter2 += 1
        print("### owo : ", counter, "### density : % ", (counter / counter2) * 100, "### açılan sandık : ", counter)

    else:
        break

bbox_file.close()
fid.close()