import cv2
import imageio
import numpy as np

from skimage.transform import resize
from sklearn.datasets import fetch_olivetti_faces
import matplotlib.pyplot as plt
from srgan import SRGAN
from util import plot_test_images
import os
from PIL import ImageFilter
from PIL import Image
from skimage.filters import gaussian




token = "\\" #for windows
results_path = "C:\\Users\\Geralt\\PycharmProjects\\FinalYearProject\\IMSGAN\\Results"
test_images_path = results_path+token+"Test_Images"
test_results_path = results_path+token+"Test_Results"
test_weights_path = results_path+token+"Test_Weights"

# Get model names to test from weights directory
model_names = os.listdir(test_weights_path)


# Test image paths
test_images = []

for image in sorted(os.listdir(test_images_path), key=lambda x: int(x.split(".")[0])):
    if image.endswith(".jpg") and image.endswith("24.jpg"):  #  -------------------------------> Alınacak imageler
        test_images.append(test_images_path+token+image)

# Create SRGAN
gan = SRGAN()

# Create result path if it does not exist
if not os.path.exists(test_results_path):
    os.makedirs(test_results_path)

for model_name in model_names:

    # SADECE BELİRLİ BİR MODEL İÇİN SONUÇ ALMAK GEREKİRSE AÇIK KALSIN, HEPSİ İÇİN SONUÇ İSTENİRSE KAPAT!
    if model_name != "Model_3":
        continue

    model_path = test_weights_path+token+model_name

    # Create model result path if it does not exist
    model_result_path = test_results_path+token+model_name
    if not os.path.exists(model_result_path):
        os.makedirs(model_result_path)


    files = os.listdir(model_path)


    # Generator weight paths
    generator_weights = []

    for filename in os.listdir(model_path):
        if filename.endswith("generator.h5"): #and filename.endswith("0000_generator.h5"): #(filename.endswith("10000_generator.h5") or filename.endswith("20000_generator.h5") or filename.endswith("30000_generator.h5") or filename.endswith("40000_generator.h5") or filename.endswith("50000_generator.h5"))    : # ----------------------------------------------------> Alınacak weightler
            generator_weights.append(model_path+token+filename)



    for file_path in generator_weights:
        gan.load_weights(file_path, file_path.replace("generator", "discriminator"))

        # reading high resolution image
        for img_path in test_images:
            img_hr = imageio.imread(img_path).astype(np.float) / 127.5 - 1
            img_hr = resize(img_hr, (256, 256), mode='constant')
            # creating low resolution image
            lr_shape = (int(img_hr.shape[0]/4), int(img_hr.shape[1]/4))
            img_lr = resize(img_hr, lr_shape, mode='constant')


            img_lr = gaussian(img_lr, sigma=0.6, multichannel=True)



            # returning image from srgan
            img_sr = gan.generator.predict(np.expand_dims(img_lr, 0))

            print("Predicted for ", "Model-> ", model_name, "Epoch-> ", file_path.split(token)[-1].split("_")[0], " Image-> ", img_path.split(token)[-1])



            img_sr = np.squeeze(img_sr, axis=0)

            #img_sr = gaussian(img_sr, sigma=0.9, multichannel=True)





            images = {
                'Low Resolution': img_lr, 'SRGAN': img_sr, 'Original': img_hr
            }

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            for i, (title, img) in enumerate(images.items()):
                axes[i].imshow(0.5 * img + 0.5)
                axes[i].set_title(title)
                axes[i].axis('off')



            # Create image result path if it does not exist
            image_result_path = model_result_path+token+img_path.split(token)[-1].split(".")[0]
            if not os.path.exists(image_result_path):
                os.makedirs(image_result_path)




            #plt.savefig(save_path+token+model_name+"_epoch_"+file_path.split(token)[-1].split("_")[0]+"_img_"+img_path.split(token)[-1])
            plt.savefig(image_result_path + token + "epoch_" + file_path.split(token)[-1].split("_")[0]) # +"-BLUR")
            plt.close()

            # save created sr image



            img_sr = (img_sr+1)*127.5
            img_sr = img_sr.astype(np.uint8)
            img_sr = cv2.cvtColor(img_sr, cv2.COLOR_RGB2BGR)
            cv2.imwrite(image_result_path + token + "epoch_" + file_path.split(token)[-1].split("_")[0] + "_SR.jpg", img_sr)

            """
            img_hr = (img_hr+1)*127.5
            img_hr = img_hr.astype(np.uint8)
            img_hr = cv2.cvtColor(img_hr, cv2.COLOR_RGB2BGR)
            cv2.imwrite(image_result_path + token + "epoch_" + file_path.split(token)[-1].split("_")[0] + "_HR.jpg", img_hr)
            """






