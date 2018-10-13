import imageio
import numpy as np

from skimage.transform import resize
from sklearn.datasets import fetch_olivetti_faces
import matplotlib.pyplot as plt
from srgan import SRGAN
from util import plot_test_images

gan = SRGAN()

gan.load_weights('./data/weights/imagenet_9_generator.h5', './data/weights/imagenet_9_discriminator.h5')

# Load image & scale it
img_hr = imageio.imread("./data/sample.jpg").astype(np.float) / 127.5 - 1

# Create a low-resolution version of it
lr_shape = (int(img_hr.shape[0]/4), int(img_hr.shape[1]/4))
img_lr = resize(img_hr, lr_shape, mode='constant')

# Predict high-resolution version (add batch dimension to image)
img_sr = gan.generator.predict(np.expand_dims(img_lr, 0))

# Remove batch dimension
img_sr = np.squeeze(img_sr, axis=0)

# Images and titles
images = {
    'Low Resolution': img_lr, 'SRGAN': img_sr, 'Original': img_hr
}

# Plot the images. Note: rescaling and using squeeze since we are getting batches of size 1
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, (title, img) in enumerate(images.items()):
    axes[i].imshow(0.5 * img + 0.5)
    axes[i].set_title(title)
    axes[i].axis('off')
plt.show()