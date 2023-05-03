#the objective here is to distill the cellpose model we trained into a smaller unet

#TODO: import the cellpose model we trained
#TODO: make the cellpose model predict on some data, binarise the predicted masks
#TODO: use that data to train a U-Net model with hyperparameters found on WandB, will have to find the correct activation function
#TODO: evaluate that model with IoU and Dice coeff
#TODO: get the instance cluste separation function
#TODO: get the wall time for both models to segment 100 images, where our trained U-Net uses the instance cluster separation function as well before giving its output
#TODO: get the memory usage for both models to segment 100 images

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
import torch.nn as nn
import torch.nn.functional as F
from skimage import measure
import matplotlib.pyplot as plt

#import data function
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

#get cellpose predictions
def get_cellpose_predictions(model,images,binary=True):
    logger = io.logger_setup()
    predictions = model.eval(images, batch_size=1, channels=[0,0], diameter=model.diam_labels)[0]
    if binary:
        predictions = [np.where(mask>0,1,0) for mask in predictions]
    return predictions

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, activation_fn, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = activation_fn(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE

def train_epoch(model, train_loader, test_loader, loss_fn, activation_fn, optimiser):
    model.train()

    #get train loss
    total_train_loss_per_epoch = 0
    for i, (x, y) in enumerate(train_loader):
        #x = x.copy()
        #y = y.copy()
        
        x = x.type(torch.float32)
        y = y.type(torch.float32)
        (x,y) = (x.to('cuda:0'), y.to('cuda:0')) # sending the data to the device (cpu or GPU)
        x = x.unsqueeze(1)
        pred = model(x)# make a prediction
        loss = loss_fn(pred, y, activation_fn) # calculate the loss of that prediction
        optimiser.zero_grad() # zero out the accumulated gradients
        loss.backward() # backpropagate the loss
        optimiser.step() # update model parameters
        total_train_loss_per_epoch += loss.detach().item()

    total_train_loss_per_epoch /= len(train_loader)
   
    #get test loss
    total_test_loss_per_epoch = 0
    total_dice = 0
    with torch.no_grad():
        for images, cellprobs in test_loader:
            #images = images.copy()
            #cellprobs = cellprobs.copy()
            
            images = images.to('cuda:0')
            cellprobs = cellprobs.to('cuda:0')

            images = torch.unsqueeze(images,1)
            cellprobs = torch.unsqueeze(cellprobs,1)
            cellprobs = cellprobs.to(torch.float32)
            outputs = model(images)

            #outputs = activation_fn(outputs)
            loss = loss_fn(outputs, cellprobs, activation_fn)
            total_test_loss_per_epoch += loss.item()

            #calculate dice score
            outputs = activation_fn(outputs)
            outputs = torch.where(outputs>0.5,1.0,0.0)
            outputs = outputs.view(-1)
            cellprobs = cellprobs.view(-1)
            intersection = (outputs * cellprobs).sum()  
            dice = (2.*intersection+1)/(outputs.sum() + cellprobs.sum()+1)  
            total_dice += dice.item()
            
    total_test_loss_per_epoch /= len(test_loader)
    total_dice /= len(test_loader)

    return total_dice

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

    #import the 02 images and masks
    images, masks = get_data("c:\\Users\\rz200\\Documents\\development\\distillCellSegTrack\\" + 'datasets/Fluo-N2DH-GOWT1/',num_imgs=92)

    #split the images and masks into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(images, masks, test_size=0.2, random_state=42)
    print('split data')

    #make the cellpose model predict on the X_train and X_test images
    cellpose_model = models.CellposeModel(gpu=core.use_gpu(), pretrained_model='/Users/rz200/Documents/development/distillCellSegTrack/segmentation/train_dir/models/cellpose_trained_model_GOWT1')
    y_train_cp = get_cellpose_predictions(cellpose_model,X_train,binary=True)
    y_test_cp = get_cellpose_predictions(cellpose_model,X_test,binary=True)
    print('got cellpose predictions')

    train_model = True
    if train_model:
        #augment the data
        for i in range(len(X_train)):
            X_train.append(np.rot90(X_train[i],2))
            y_train_cp.append(np.rot90(y_train_cp[i],2))

        #to torch
        X_train_torch = np.array([torch.from_numpy(np.array(x)) for x in X_train])
        y_train_cp_torch = np.array([torch.from_numpy(np.array(x)) for x in y_train_cp])
        X_test_torch = np.array([torch.from_numpy(np.array(x)) for x in X_test])
        y_test_cp_torch = np.array([torch.from_numpy(np.array(x)) for x in y_test_cp])

        #get data loaders
        train_loader = torch.utils.data.DataLoader(list(zip(X_train_torch,y_train_cp_torch)), batch_size=16, shuffle=True)
        test_loader = torch.utils.data.DataLoader(list(zip(X_test_torch,y_test_cp_torch)), batch_size=16, shuffle=True)

        #get the model and configurations
        model = UNet()
        model = model.to('cuda:0')
        loss_fn = DiceBCELoss()
        optimiser = torch.optim.RMSprop(model.parameters(), lr=0.001)

        #train the model
        for epoch in range(30):
            dice = train_epoch(model, train_loader, test_loader, loss_fn, torch.sigmoid, optimiser)
            print(dice)

        #save the model
        torch.save(model.state_dict(), 'c:\\Users\\rz200\\Documents\\development\\distillCellSegTrack\\segmentation\\train_dir\\models\\unet_trained_model')

    #get the instance cluster separation function
    #instance = measure.label(y_train_cp[0], connectivity=2)

    #import the trained model
    model = UNet()
    model = model.to('cuda:0')
    model.load_state_dict(torch.load('c:\\Users\\rz200\\Documents\\development\\distillCellSegTrack\\segmentation\\train_dir\\models\\unet_trained_model'))

    #get predictions on test_cp
    predicted_masks = []
    for x in X_test:
        prediction = model(torch.unsqueeze(torch.from_numpy(np.array(x)),0).to('cuda:0').to(torch.float32).unsqueeze(1))
        prediction = torch.sigmoid(prediction)
        prediction = torch.where(prediction>0.5,1.0,0.0)
        prediction = prediction.squeeze(0).squeeze(0).cpu().detach().numpy()
        predicted_masks.append(prediction)

    #make predictions on the test_cp and get the IoU and Dice scores
    #get IoU and dice coeff
    y_test_cp = np.array(y_test_cp)
    IoU = get_IoU(predicted_masks,y_test_cp)
    dice = get_dice(predicted_masks,y_test_cp)
    print('IoU: ', IoU)
    print('Dice: ', dice)

    for i in range(19):
        plt.subplot(3,1,1)
        plt.imshow(X_test[i])

        plt.subplot(3,1,2)
        plt.imshow(y_test_cp[i])

        plt.subplot(3,1,3)
        plt.imshow(predicted_masks[i])

        plt.show()
    #we will do the evaluations on a different py file to just train here
    #evaluate the U-Net model on the X_test images and y_test_cp masks
