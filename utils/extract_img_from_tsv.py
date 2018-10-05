import os
import base64
import struct
import cv2


min_size = 256
counter = 0
counter2 = 0

size_input_img = 64 #input image size
size_output_img = 256 # output image size

dir_sep = "/" #Dir seperator for windows use \\
img_extension = ".jpg" #Image File extension

fid = open("/home/student_1/IMSGAN/zipdataset/FaceImageCroppedWithAlignment.tsv", encoding='utf-8')

base_path = '/home/student_1/IMSGAN/zipdataset/base' # Tsv extract path
dir_all_images = "/home/student_1/IMSGAN/dataset" #Dataset directory has 2 subfolders : 64 and 256
dir_input_images = dir_all_images+dir_sep+str(size_input_img)  #network input subdir named 64
dir_output_images = dir_all_images+dir_sep+str(size_output_img) #network output subdir named 256

try:
    os.mkdir(dir_input_images)
    os.mkdir(dir_output_images)
except:
    pass

if not os.path.exists(base_path):
    os.mkdir(base_path)

bbox_file = open(base_path + '/bboxes.txt', 'w')

while True:
    line = fid.readline()
    if line:
        data_info = line.split('\t')
        # 0: Freebase MID (unique key for each entity)
        # 1: ImageSearchRank
        # 4: FaceID
        # 5: bbox
        # 6: img_data
        filename = data_info[0] + dir_sep + data_info[1] + "_" + data_info[4] + img_extension
        bbox = struct.unpack('ffff', base64.b64decode(data_info[5]))
        # bbox = struct.unpack('ffff', data_info[5].decode("base64"))
        bbox_file.write(filename + " " + (" ".join(str(bbox_value) for bbox_value in bbox)) + "\n")

        img_data = base64.b64decode(data_info[6])

        output_file_path = base_path + dir_sep + filename
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

        #Shape control of given images to fit minimum requirements for network
        if height >= min_size and width >= min_size and channel == 3:
            counter += 1
            #Resize as input image dim
            im1 = cv2.resize(image, (size_input_img, size_input_img))
            #Resize as output image dim
            im2 = cv2.resize(image, (size_output_img, size_output_img))
            #Save as input image dim
            cv2.imwrite(dir_input_images + dir_sep + str(counter)+img_extension, im1)
            #Save as input image dim
            cv2.imwrite(dir_output_images + dir_sep + str(counter)+img_extension, im2)
            #Remove the useless original image
            os.remove(output_file_path)
        else:
            #Remove image that not fits control condition
            os.remove(output_file_path)

        counter2 += 1
        print("### owo : ", counter, "### density : % ", (counter / counter2) * 100, "### açılan sandık : ", counter2)

    else:
        break

bbox_file.close()
fid.close()
