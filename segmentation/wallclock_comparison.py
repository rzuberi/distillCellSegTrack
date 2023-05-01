#here we compare the walltime and memory usage of Cellpose vs our distilled U-Net model
#we will combine the test images of both datasets to make sure the models have not seen these images, and the random seed is still 42

import numpy as np
from os import listdir
from os.path import isfile, join
import tifffile
from cellpose import models, io, core
import time
from sklearn.model_selection import train_test_split
from statistics import mean
from u_net import UNet
import torch
from skimage import measure

def get_data(path, num_imgs=4, set='01'):

    images_path = path + set + '/'
    onlyfiles = [f for f in listdir(images_path) if isfile(join(images_path, f))]
    onlyfiles.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    if num_imgs > len(onlyfiles): num_imgs = len(onlyfiles)
    images = [np.squeeze(tifffile.imread(images_path +  onlyfiles[i])) for i in range(num_imgs)]
    images = [(image-np.min(image))/(np.max(image)-np.min(image)) for image in images]
    
    masks_path = path + set + '_GT/TRA/'
    onlyfiles = [f for f in listdir(masks_path) if isfile(join(masks_path, f))][1:]
    onlyfiles.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    if num_imgs > len(onlyfiles): num_imgs = len(onlyfiles)
    masks = [np.squeeze(tifffile.imread(masks_path +  onlyfiles[i])) for i in range(num_imgs)]

    return images, masks

if __name__ == '__main__':
    #get the 01 images and masks
    #split them to get the test images
    images_01, masks_01 = get_data("c:\\Users\\rz200\\Documents\\development\\distillCellSegTrack\\" + 'datasets/Fluo-N2DL-HeLa/', num_imgs=92, set='01')
    images_01_train, images_01_test, masks_01_train, masks_01_test = train_test_split(images_01, masks_01, test_size=0.2, random_state=42)

    #get the 02 images and masks
    #split them to get the test images
    images_02, masks_02 = get_data("c:\\Users\\rz200\\Documents\\development\\distillCellSegTrack\\" + 'datasets/Fluo-N2DL-HeLa/', num_imgs=92, set='02')
    images_02_train, images_02_test, masks_02_train, masks_02_test = train_test_split(images_02, masks_02, test_size=0.2, random_state=42)

    #combine the test images into one list
    test_images = images_01_test + images_02_test

    #get the cellpose model
    #run the cellpose model on the test images
    start = time.time()
    cellpose_model = models.CellposeModel(gpu=core.use_gpu(), pretrained_model='segmentation/train_dir/models/cellpose_trained_model')
    cellpose_predicted_masks = cellpose_model.eval(test_images, batch_size=1, channels=[0,0], diameter=cellpose_model.diam_labels)[0]
    print("cellpose time: ", time.time()-start)

    #get the distilled unet model
    #run the unet model on the test images
    #run the instance segmentation as well (only fair since cellpose also returns the instance segmentation)
    start = time.time()
    model = UNet()
    model = model.to('cuda:0')
    model.load_state_dict(torch.load('c:\\Users\\rz200\\Documents\\development\\distillCellSegTrack\\segmentation\\train_dir\\models\\unet_trained_model'))
    distilled_predicted_masks = []
    for test_image in test_images:
        test_image = torch.from_numpy(test_image).float().unsqueeze(0).unsqueeze(0).to('cuda:0')
        predicted_binary_mask = model(test_image).squeeze(0).squeeze(0).cpu().detach().numpy()
        predicted_instance_mask = measure.label(predicted_binary_mask, connectivity=2)
        distilled_predicted_masks.append(predicted_instance_mask)
    print("distilled time: ", time.time()-start)

    

    

    #compare the timings (seconds)
    #cellpose time:  48.18089938163757
    #distilled time:  1.1167442798614502

    #then we will compare memory usage