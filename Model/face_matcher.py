import imageio
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
import matplotlib.pyplot as plt
import scipy.misc

# CONSTANTS OF MATCHING ALGORITHM.
hr_shape = 256
lr_shape = 64
k_value = 3
epsilon = 0.35 # A threshold value for verification that used with cosine similarity measure.
dir_feature_database = '/Users/Macbook/Desktop/FinalYearProject/IMSGAN/Model/database/data.p'
# VGG Face Network Definition
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

# A function that make images ready for input by converting np array
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(hr_shape, hr_shape))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    # print(img[0:100])
    return img

# A function that calculates cosine similarity measure between two vectors
def findCosineSimilarity(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

# A function that calculates Euclidean Distance between two vectors
def findEuclideanDistance(source_representation, test_representation):
    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance

# A function that creates features of all images from main database and saves feature maps of all images to file. It helps to warm starting of matching algorithm.
def create_database_features(database_dir):
    images = sorted(os.listdir(database_dir))
    images= [image for image in images if image.endswith(".jpg")]
    feature_map = {}
    for image in images:
        feature_map[database_dir+"/"+image] = vgg_face_descriptor.predict(preprocess_image(database_dir+"/"+image))[0, :]

    with open(dir_feature_database, 'wb') as fp:
        pickle.dump(feature_map, fp, protocol=pickle.HIGHEST_PROTOCOL)
    print("Features of image database saved into file...")

# A function that loads feature maps of all images from main database and returns them as an dictionary.
def load_database(pkl_database_dir):
    with open(pkl_database_dir, 'rb') as fp:
        database = pickle.load(fp)
    print("Database loaded to memory...")
    return database

# A function that returns the most similar image filenames of given image file.
def matcher(feature_database,test_img_filename):
    print("Test image : ",test_img_filename.split('/')[-1])
    scores = {}
    euclideans = {}
    test_img_features = vgg_face_descriptor.predict(preprocess_image(test_img_filename))[0, :]
    for key,value in feature_database.items():
        score = findCosineSimilarity(test_img_features,value)
        euclidean = findEuclideanDistance(test_img_features,value)
        scores[key] = score
        euclideans[key] = euclidean
    sorted_scores = dict([(k, scores[k]) for k in sorted(scores, key=scores.get)])
    sorted_euclideans = dict([(k, euclideans[k]) for k in sorted(euclideans, key=euclideans.get)])
    # print(sorted_scores)
    # for key,value in sorted_scores.items():
    #     print(key.split('/')[-1]," ===> ",value)
    return sorted_scores,sorted_euclideans

def plotter( euclidean_scores, cosine_scores, k_value, test_img_dir):
    # Load image & scale it
    image_dirs = []
    results = []
    image_dirs.append(test_img_dir)
    results.append([0,0])
    similar_filenames = list(cosine_scores.keys())
    cosine_values = list(cosine_scores.values())
    euclidean_values = list(euclidean_scores.values())

    for i in range(0,k_value):
        image_dirs.append(similar_filenames[i])
        results.append([cosine_values[i],euclidean_values[i]])

    fig, axes = plt.subplots(1, k_value+1, figsize=((k_value+1)*5, 5))
    for i in range(0,k_value+1):
        img = imageio.imread(image_dirs[i]).astype(np.float) / 127.5 - 1
        axes[i].imshow(0.5 * img + 0.5)
        axes[i].set_title(image_dirs[i].split("/")[-1])
        xlabel = "Cosine : "+str(results[i][0])[:6]+" Euclidean : "+str(results[i][1])[:6]
        print(xlabel)
        axes[i].set_xlabel(xlabel)
        axes[i].axis('on')
    plt.show()




database_path= "/Users/Macbook/Desktop/FinalYearProject/Results/Test_Images"

model = vgg_face('/Users/Macbook/Desktop/FinalYearProject/IMSGAN/vgg_face_weights.h5')
# model.summary()

vgg_face_descriptor = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)

feature_database = load_database(dir_feature_database)
# create_database_features(database_dir=database_path)


test_path = "/Users/Macbook/Desktop/FinalYearProject/Results/Test_Images/"
test2_path = "/Users/Macbook/Desktop/FinalYearProject/Results/test/"
test2_filenames = [item for item in sorted(os.listdir(test2_path)) if item.endswith(".jpg")]
print(test2_filenames)
img_1 = "29.jpg"
img_2 = "24.jpg"

# scipy.misc.imsave('outfile.jpg', image_array)
for fname in test2_filenames:
    cosine_scores,euclidean_scores = matcher(feature_database,test2_path+fname)
    plotter(euclidean_scores,cosine_scores,k_value,test2_path+fname)