#Test cellpose without training a model


import numpy as np
from os import listdir
from os.path import isfile, join
import tifffile
from cellpose import models, io, core
import time
from sklearn.model_selection import train_test_split
from statistics import mean
import torch
from ptflops import get_model_complexity_info

def get_files(path,normalise=False,remove_txt=False):
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]

    if remove_txt:
        onlyfiles = [val for val in onlyfiles if not val.endswith(".txt")]

    onlyfiles.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    #if num_imgs > len(onlyfiles): num_imgs = len(onlyfiles)
    files = [np.squeeze(tifffile.imread(path +  onlyfiles[i])) for i in range(len(onlyfiles))]
    
    if normalise:
        files = [(image-np.min(image))/(np.max(image)-np.min(image)) for image in files]
    
    return files
    
def get_data(path, set='01',normalise_images=True):

    if len(set) == 2: #set 01 or set 02
        images_path = path + set + '/'
        images = get_files(images_path,normalise=normalise_images)
        masks_path = path + set + '_GT/TRA/'
        masks = get_files(masks_path,remove_txt=True)
    elif set == '0102': #both sets
        images_path = path + '01/'
        images_01 = get_files(images_path,normalise=normalise_images)
        images_path = path + '02/'
        images_02 = get_files(images_path,normalise=normalise_images)
        images = images_01 + images_02

        masks_path = path + '01_GT/TRA/'
        masks_01 = get_files(masks_path,remove_txt=True)
        masks_path = path + '02_GT/TRA/'
        masks_02 = get_files(masks_path,remove_txt=True)
        masks = masks_01 + masks_02
    else:
        images = []
        masks = []

    return images, masks

def get_IoU(predicted_masks,gt_masks,return_list=False):
    intersection_unions = []
    for i in range(len(predicted_masks)):
        intersection = np.logical_and(predicted_masks[i], gt_masks[i]).sum()
        union = np.logical_or(predicted_masks[i], gt_masks[i]).sum()
        intersection_unions.append(intersection/union)
    if return_list:
        return intersection_unions
    return mean(intersection_unions)

#function to get pixel wise accuracy
def get_accuracy(predicted_masks,gt_masks,return_list=False):
    accuracies = []
    for i in range(len(predicted_masks)):
        accuracies.append(np.mean(predicted_masks[i] == gt_masks[i]))
    if return_list:
        return accuracies
    return mean(accuracies)

if __name__ == '__main__':
    logger = io.logger_setup()
    model = models.CellposeModel(gpu=core.use_gpu(), model_type='cyto', device=torch.device('cuda:0'))
    unet = models.UnetModel(gpu=core.use_gpu()).network

    images, masks = get_data('datasets/Fluo-N2DH-GOWT1/', set='0102', normalise_images=False)
    X_train, X_test, y_train, y_test = train_test_split(images, masks, test_size=0.2, random_state=42)

    predicted_masks = model.eval(X_test, batch_size=1, channels=[0,0], diameter=model.diam_labels)[0]
    print(len(predicted_masks))

    predicted_masks = [np.where(mask>0,1,0) for mask in predicted_masks]
    y_test_binary = [np.where(mask>0,1,0) for mask in y_test]

    IoU = get_IoU(predicted_masks,y_test_binary, return_list=True)
    acc = get_accuracy(predicted_masks,y_test_binary,   return_list=True)
    #translate this next bit to code
    #Mean IoU:
    #Max IoU:
    #Min IoU:
    #Mean Pixel-wise:
    #Max Pixel-wise:
    #Min Pixel-wise:
    print('mean IoU',mean(IoU))
    print('max IoU',max(IoU))
    print('min IoU',min(IoU))
    print('mean pixel-wise',mean(acc))
    print('max pixel-wise',max(acc))
    print('min pixel-wise',min(acc))