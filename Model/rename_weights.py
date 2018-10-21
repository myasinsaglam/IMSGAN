import os
import numpy as np

weight_path = "C:\\Users\\Geralt\\PycharmProjects\\FinalYearProject\\IMSGAN\\Results\\Test_Weights\\Model_1"
token = "\\"
weight_names = os.listdir(weight_path)

for weight_name in weight_names:
    full_path = weight_path + token + weight_name

    if weight_name.endswith(".h5"):
        if np.size(weight_name.split(token)[-1].split("_")) > 2:

            final_old = weight_path+token+weight_name
            final_new = weight_path+token+weight_name.split("_")[-2] + "_" + weight_name.split("_")[-1]

            #os.rename(weight_name, weight_name.split("_")[-2] + "_" + weight_name.split("_")[-1])
            os.rename(final_old, final_new)
