#Train a cellpose model with the Fluo-N2DL-HeLa dataset
#We will then evaluate that model in another file

import numpy as np
from os import listdir
from os.path import isfile, join
import tifffile
from cellpose import models, io, core
import time
from sklearn.model_selection import train_test_split
from statistics import mean

def get_data(path, num_imgs=4):

    images_path = path + '01/'
    onlyfiles = [f for f in listdir(images_path) if isfile(join(images_path, f))]
    onlyfiles.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    if num_imgs > len(onlyfiles): num_imgs = len(onlyfiles)
    images = [np.squeeze(tifffile.imread(images_path +  onlyfiles[i])) for i in range(num_imgs)]
    images = [(image-np.min(image))/(np.max(image)-np.min(image)) for image in images]
    
    masks_path = path + '01_GT/TRA/'
    onlyfiles = [f for f in listdir(masks_path) if isfile(join(masks_path, f))][1:]
    onlyfiles.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    if num_imgs > len(onlyfiles): num_imgs = len(onlyfiles)
    masks = [np.squeeze(tifffile.imread(masks_path +  onlyfiles[i])) for i in range(num_imgs)]
    #binarise the masks
    #masks = [np.where(mask>0,1,0) for mask in masks]

    return images, masks

def train_model(images,masks,n_epochs,model_name):
    X_train, X_test, y_train, y_test = train_test_split(images, masks, test_size=0.2, random_state=42)

    print(len(X_train), len(X_test), len(y_train), len(y_test))
    
    logger = io.logger_setup()
    model = models.CellposeModel(gpu=core.use_gpu(), model_type='cyto')
    new_model_path = model.train(X_train, y_train, 
                              test_data=X_test,
                              test_labels=y_test,
                              channels=[0,0], 
                              save_path='train_dir', 
                              n_epochs=n_epochs,
                              learning_rate=0.1, 
                              weight_decay=0.0001, 
                              nimg_per_epoch=8,
                              model_name=model_name)

def get_IoU(predicted_masks,gt_masks):
    intersection_unions = []
    for i in range(len(predicted_masks)):
        intersection = np.logical_and(predicted_masks[i], gt_masks[i]).sum()
        union = np.logical_or(predicted_masks[i], gt_masks[i]).sum()
        intersection_unions.append(intersection/union)
    return mean(intersection_unions)

def get_dice(predicted_masks,gt_masks):
    dices = []
    for i in range(len(predicted_masks)):
        intersection = np.logical_and(predicted_masks[i], gt_masks[i]).sum()
        dice = (2*intersection)/(predicted_masks[i].sum() + gt_masks[i].sum())
        dices.append(dice)
    return mean(dices)

if __name__ == '__main__':
    #TODO: put the training in a function
    #TODO: make an evaluation function to import the model and test it on the test data with dice coeff and IoU
    #TODO: import the U-Net model, train it on the same data, use the hyperparameters from WandB, evaluate it
    #TODO: get the wall time for both models to segment 100 images
    #TODO: get the memory usage for both models to segment 100 images
    #TODO: get the tracking function
    
    images, masks = get_data('datasets/Fluo-N2DL-HeLa/', num_imgs=92)
    print(len(images))
    print(len(masks))
    print(type(images),type(masks))
    print(images[0].shape[0])
    print(masks[0].shape[0])
    train_model(images,masks,500,'cellpose_trained_model')

    X_train, X_test, y_train, y_test = train_test_split(images, masks, test_size=0.2, random_state=42)

    #import our trained cellpose model
    model = models.CellposeModel(gpu=core.use_gpu(), pretrained_model='segmentation/train_dir/models/cellpose_trained_model')
    predicted_masks = model.eval(X_test, batch_size=1, channels=[0,0], diameter=model.diam_labels)[0]
    #binarise the predicted masks
    predicted_masks = [np.where(mask>0,1,0) for mask in predicted_masks]
    y_test_binary = [np.where(mask>0,1,0) for mask in y_test]

    print(np.unique(predicted_masks,return_counts=True))
    print(np.unique(y_test_binary,return_counts=True))

    
    #get IoU and dice coeff
    IoU = get_IoU(predicted_masks,y_test_binary)
    dice = get_dice(predicted_masks,y_test_binary)
    print('IoU: ', IoU)
    print('Dice: ', dice)