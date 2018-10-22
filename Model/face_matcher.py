from keras.models import Model, Sequential
from keras.layers import Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from PIL import Image
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
import matplotlib.pyplot as plt
from keras.models import model_from_json
import os
import pickle


hr_shape = 256
lr_shape = 64

def vgg_face(weights_path):
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(hr_shape, hr_shape, 3)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Convolution2D(4096, (7, 7), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(4096, (1, 1), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(2622, (1, 1)))
    model.add(Flatten())
    model.add(Activation('softmax'))
    model.load_weights(weights_path)
    # model.summary()
    return model

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(hr_shape, hr_shape))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

def findCosineSimilarity(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

def findEuclideanDistance(source_representation, test_representation):
    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance

def create_database_features(database_dir):
    images = sorted(os.listdir(database_dir))
    images= [image for image in images if image.endswith(".jpg")]
    feature_map = {}
    for image in images:
        feature_map[database_dir+"/"+image] = vgg_face_descriptor.predict(preprocess_image(database_dir+"/"+image))[0, :]

    with open('/Users/Macbook/Desktop/FinalYearProject/IMSGAN/Model/database/data.p', 'wb') as fp:
        pickle.dump(feature_map, fp, protocol=pickle.HIGHEST_PROTOCOL)
    print("Features of image database saved into file...")

def load_database(pkl_database_dir):
    with open(pkl_database_dir, 'rb') as fp:
        database = pickle.load(fp)
    print("Database loaded to memory...")
    return database




database_path= "/Users/Macbook/Desktop/FinalYearProject/Results/Test_Images"

model = vgg_face('/Users/Macbook/Desktop/FinalYearProject/IMSGAN/vgg_face_weights.h5')
# model.summary()
vgg_face_descriptor = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)

feature_database = load_database("/Users/Macbook/Desktop/FinalYearProject/IMSGAN/Model/database/data.p")
create_database_features(database_dir=database_path)

epsilon = 0.35

test_path = "/Users/Macbook/Desktop/FinalYearProject/Results/Test_Images/"
img_1 = "29.jpg"
img_2 = "24.jpg"


def matcher(feature_database,test_img_filename):
    print("Test image : ",test_img_filename.split('/')[-1])
    scores = {}
    test_img_features = vgg_face_descriptor.predict(preprocess_image(test_img_filename))[0, :]
    for key,value in feature_database.items():
        score = findCosineSimilarity(test_img_features,value)
        scores[key] = score
    sorted_scores = dict([(k, scores[k]) for k in sorted(scores, key=scores.get)])
    print(sorted_scores)
    for key,value in sorted_scores.items():
        print(key.split('/')[-1]," ===> ",value)


matcher(feature_database,test_path+img_1)

# def verifyFace(img1, img2):
#     img1_representation = vgg_face_descriptor.predict(
#         preprocess_image(test_path+img1))[0, :]
#     print(img1_representation)
#     img2_representation = vgg_face_descriptor.predict(
#         preprocess_image(test_path+img2))[0, :]
#
#     cosine_similarity = findCosineSimilarity(img1_representation, img2_representation)
#     euclidean_distance = findEuclideanDistance(img1_representation, img2_representation)
#
#     print("Cosine similarity: ", cosine_similarity)
#     print("Euclidean distance: ", euclidean_distance)
#
#     if (cosine_similarity < epsilon):
#         print("verified... they are same person")
#     else:
#         print("unverified! they are not same person!")
#
#     f = plt.figure()
#     f.add_subplot(1, 2, 1)
#     plt.imshow(image.load_img(test_path+img1))
#     plt.xticks([]);
#     plt.yticks([])
#     f.add_subplot(1, 2, 2)
#     plt.imshow(image.load_img(test_path+img2))
#     plt.xticks([]);
#     plt.yticks([])
#     plt.show(block=True)
#     print("-----------------------------------------")
#
# verifyFace(img_1,img_2)