#here we compare the memory usage

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

    #only take 1 image
    test_image = images_01_test[0]

    #get the cellpose model
    #run the cellpose model on the test images

    

    #cellpose_model = models.CellposeModel(gpu=core.use_gpu(), pretrained_model='segmentation/train_dir/models/cellpose_trained_model')
    #cellpose_predicted_masks = cellpose_model.eval(test_image, batch_size=1, channels=[0,0], diameter=cellpose_model.diam_labels)[0]

    


    #get the distilled unet model
    #run the unet model on the test images
    #run the instance segmentation as well (only fair since cellpose also returns the instance segmentation)

    print('Memory Usage:')
    allocated = torch.cuda.memory_allocated(0)/1024**3
    print('Allocated:', allocated)
    cached = torch.cuda.memory_reserved(0)/1024**3
    print('Cached:   ', cached)

    model = UNet()
    model = model.to('cuda:0')
    model.load_state_dict(torch.load('c:\\Users\\rz200\\Documents\\development\\distillCellSegTrack\\segmentation\\train_dir\\models\\unet_trained_model'))

    print('Memory Differences:')
    allocated_diff = torch.cuda.memory_allocated(0)/1024**3 - allocated
    print('Allocated diff:', allocated_diff)
    cached_diff = torch.cuda.memory_reserved(0)/1024**3 - cached
    print('Cached diff:   ', cached_diff)

    #test_image = torch.from_numpy(test_image).float().unsqueeze(0).unsqueeze(0).to('cuda:0')
    #predicted_binary_mask = model(test_image).squeeze(0).squeeze(0).cpu().detach().numpy()

    
    #predicted_instance_mask = measure.label(predicted_binary_mask, connectivity=2)


    #CELLPOSE
        #Memory Usage:
        #Allocated: 0.0
        #Cached:    0.0
        #Memory Differences after predicting 1 image:
        #Allocated diff: 0.02468109130859375
        #Cached diff:    0.15625

    #UNET
        #Memory Usage:
        #Allocated: 0.0
        #Cached:    0.0
        #Memory Differences after predicting 1 image:
        #Allocated diff: 0.0004405975341796875
        #Cached diff:    0.001953125
    