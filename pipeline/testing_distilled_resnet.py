from resnet_archi import CPnet
import torch
import os
import numpy as np
from distilled_resnet import run_cp, make_prediction
import matplotlib.pyplot as plt

if __name__ == '__main__':

    #importing the images
    image_folder = "/Users/rz200/Documents/development/distillCellSegTrack/pipeline/saved_cell_images_1237"
    combined_images = []
    i = 0
    for filename in os.listdir(image_folder):
        if filename.endswith('.npy'):
            image_path = os.path.join(image_folder,filename)
            numpy_image = np.load(image_path)
            combined_images.append(numpy_image)
            i += 1
        if i == 1: 
            break

    model = CPnet(nbase=[1,32], nout=3, sz=3,
                residual_on=True, style_on=True,
                concatenation=False, mkldnn=False)
    model.load_model("/Users/rz200/Documents/development/distillCellSegTrack/pipeline/resnet_nuc_32", device=torch.device('cuda:0'))
    print(combined_images[0][1].shape)
    prediction = make_prediction(combined_images[0][1],model, 'cuda:0', 'nuclei')

    print(prediction)
    plt.imshow(prediction)
    plt.show()