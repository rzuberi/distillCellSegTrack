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
import torch

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

def train_model(images,masks,n_epochs,model_name):
    X_train, X_test, y_train, y_test = train_test_split(images, masks, test_size=0.2, random_state=42)

    print(len(X_train), len(X_test), len(y_train), len(y_test))

    #a list of elements that starts with 0.0005 at every element and halves every 50 elements for up to 300 elements
    learning_rates = [0.1/(3**(i//50)) for i in range(300)]
    
    
    model = models.CellposeModel(gpu=core.use_gpu(), model_type='cyto', device=torch.device('cuda:0'))
    new_model_path = model.train(X_train, y_train, 
                              test_data=X_test,
                              test_labels=y_test,
                              channels=[0,0], 
                              save_path='train_dir', 
                              n_epochs=n_epochs,
                              learning_rate=0.1,
                              weight_decay=0.0001,
                              model_name=model_name,
                              batch_size=16,
                              SGD=True)

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
    
    #images, masks = get_data('datasets/Fluo-N2DL-HeLa/', num_imgs=92)
    images, masks = get_data('datasets/Fluo-N2DH-SIM+/', set='0102', normalise_images=False)
    #images, masks = get_data('datasets/Fluo-N2DL-GOWT1/', set='0102', normalise_images=False)

    print(len(images))
    print(len(masks))
    print(type(images),type(masks))
    print(images[0].shape[0])
    print(masks[0].shape[0])
    logger = io.logger_setup()
    train_model(images,masks,400,'cellpose_trained_model_SIM_3')

    X_train, X_test, y_train, y_test = train_test_split(images, masks, test_size=0.2, random_state=42)

    #import our trained cellpose model
    #maybe we should add "model_type='cyto'" to the model before training it
    model = models.CellposeModel(gpu=core.use_gpu(), device=torch.device("cuda:0"), pretrained_model='/Users/rz200/Documents/development/distillCellSegTrack/segmentation/train_dir/models/cellpose_trained_model_SIM_2')
    predicted_masks = model.eval(X_test, batch_size=1, channels=[0,0], diameter=model.diam_labels)[0]
    print(len(predicted_masks))
    #binarise the predicted masks
    predicted_masks = [np.where(mask>0,1,0) for mask in predicted_masks]
    y_test_binary = [np.where(mask>0,1,0) for mask in y_test]

    #print(np.unique(predicted_masks,return_counts=True))
    #print(np.unique(y_test_binary,return_counts=True))

    
    #get IoU and dice coeff
    IoU = get_IoU(predicted_masks,y_test_binary)
    dice = get_dice(predicted_masks,y_test_binary)
    print('IoU: ', IoU)
    print('Dice: ', dice)