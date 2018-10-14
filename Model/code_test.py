#Loss plot

import numpy as np
import pickle
import  os
from matplotlib import pyplot as plt

path = "/Users/Macbook/Desktop/Dataset/LOSS"
filedirs = os.listdir(path)

# for filename in filedirs:
with open(path+"/1e-4_less_img_11_losses.p", "rb") as f:
    dictname = pickle.load(f)
    # print(dictname)

discriminator_loss = [item['discriminator'][0] for item in dictname]
vgg_loss = [item['generator'][2] for item in dictname]
generator_loss = [item['generator'][0] for item in dictname]
disc_gen_loss = [item['generator'][1] for item in dictname]
print(discriminator_loss)
# for elem in dictname:
    # print(elem['discriminator'][0])
fig = plt.figure()

x =0
plt.plot(np.arange(x,len(discriminator_loss)), discriminator_loss[x:], label="disc_loss")
plt.plot(np.arange(x,len(discriminator_loss)), generator_loss[x:], label="gen_loss")
plt.plot(np.arange(x,len(discriminator_loss)), vgg_loss[x:], label="vgg_loss")
plt.plot(np.arange(x,len(discriminator_loss)), disc_gen_loss[x:], label="gen_disc_loss")


# plt.plot(self.x, self.val_losses, label="val_loss")
plt.legend()
plt.show()

#1 vgg , 2 - gen , 3 -disc
