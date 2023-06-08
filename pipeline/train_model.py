#Here we train the model

from import_images import getImages
from import_model import getModel
from make_predictions import makePredictions
import numpy as np

import torch

from cellpose import resnet_torch
from cellpose import transforms
from cellpose import utils
import cv2

import time

from unet_architecture import UNet
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torchvision
import numpy as np

import matplotlib.pyplot as plt

def get_pre_activations(image,cpnet):
    rescale = cpnet.diam_mean/cpnet.diam_labels
    shape1, shape2 = image.shape[0], image.shape[1]

    x = transforms.resize_image(image, rsz=rescale,no_channels=True)
    x = x.astype(np.float32)
    x = np.expand_dims(x, axis=0)
    x = np.concatenate((x, x), axis=0)
    x = torch.from_numpy(x)
    x = x.unsqueeze(0)

    downsample = cpnet.downsample(x)

    style = cpnet.make_style(downsample[-1])
    upsample = cpnet.upsample(style, downsample, cpnet.mkldnn)

    output = cpnet.output(upsample)
    output = output.squeeze(0)
    output = output[2]
    output = output.cpu().detach().numpy().tolist()
    output = cv2.resize(np.array(output), dsize=(512, 512), interpolation=cv2.INTER_NEAREST)
    output = np.array(output)
    output = torch.from_numpy(output)

    upsample = upsample.squeeze(0)
    upsample = upsample.cpu().detach().numpy().tolist()
    for (k, image) in enumerate(upsample):
        upsample[k] = cv2.resize(np.array(image), dsize=(512, 512), interpolation=cv2.INTER_NEAREST)
    upsample = np.array(upsample)
    upsample = torch.from_numpy(upsample)

    return upsample, output

class ImageDataset(Dataset):
    def __init__(self, image, cellprob, cellmask):
        self.image = image
        self.cellprob = cellprob
        self.cellmask = cellmask

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        img = self.image[idx]
        cellprob = self.cellprob[idx]
        cellmask = self.cellmask[idx]
        return img, cellprob, cellmask
    
def augment_data(images, cellpose_upsample, cellpose_output):

    images_torch_rotated = []
    for image in images:
        images_torch_rotated.append(image)
        for i in range(3):
            image = torch.rot90(image, 1, [0, 1])
            images_torch_rotated.append(image)

    images_torch_rotated_flipped = []
    for image in images_torch_rotated:
        images_torch_rotated_flipped.append(image)
        images_torch_rotated_flipped.append(torch.flip(image, [1]))

    cp_upsamples_rotated = []
    for upsample in cellpose_upsample:
        cp_upsamples_rotated.append(upsample)
        for i in range(3):
            upsample = torch.rot90(upsample, 1, [1, 2])
            cp_upsamples_rotated.append(upsample)

    cp_upsamples_rotated_flipped = []
    for upsample in cp_upsamples_rotated:
        cp_upsamples_rotated_flipped.append(upsample)
        cp_upsamples_rotated_flipped.append(torch.flip(upsample, [2]))

    cp_outputs_rotated = []
    for output in cellpose_output:
        cp_outputs_rotated.append(output)
        for i in range(3):
            output = torch.rot90(output, 1, [0, 1])
            cp_outputs_rotated.append(output)

    cp_outputs_rotated_flipped = []
    for output in cp_outputs_rotated:
        cp_outputs_rotated_flipped.append(output)
        cp_outputs_rotated_flipped.append(torch.flip(output, [1]))

    return images_torch_rotated_flipped, cp_upsamples_rotated_flipped, cp_outputs_rotated_flipped

class KDLoss(torch.nn.Module):
    def __init__(self, alpha = 1.0, beta = 1.0, temperature=1):
        super(KDLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature

    def forward(self, y_32_pred, y_3_pred, y_32_true, y_3_true):

        y_32_pred = F.sigmoid(y_32_pred)
        y_32_true = F.sigmoid(y_32_true)
        y_32_loss = F.mse_loss(y_32_pred, y_32_true.float())

        y_3_pred = F.sigmoid(y_3_pred)
        y_3_true = F.sigmoid(y_3_true)
        y_3_loss = F.mse_loss(y_3_pred, y_3_true.float())

        loss = self.alpha * y_32_loss + self.beta * y_3_loss
        return loss
    
def trainEpoch(unet, train_loader, loss_fn, optimiser, scheduler, epoch_num):
    time_start = time.time()
    
    unet.train()

    train_loss = 0
    for image, upsample, cp_output in train_loader:
        (image,upsample,cp_output) = (image.to('cuda:0'),upsample.to('cuda:0'),cp_output.to('cuda:0')) # sending the data to the device (cpu or GPU)

        image = image.unsqueeze(1)
        y_32_pred, y_3_pred = unet(image)
        y_32_pred = y_32_pred.squeeze(1)
        y_3_pred = y_3_pred.squeeze(1)
        
        loss = loss_fn(y_32_pred, y_3_pred, upsample, cp_output) # calculate the loss of that prediction

        train_loss += loss
        optimiser.zero_grad() # zero out the accumulated gradients
        loss.backward() # backpropagate the loss
        optimiser.step() # update model parameters
        if scheduler is not None:
            scheduler.step()
    train_loss = train_loss.item()/len(train_loader)

    if epoch_num is None:
        print('Training loss: ', train_loss, 'Time: ', time.time()-time_start)
    else:
        print('Epoch ', epoch_num, 'Training loss: ', train_loss, 'Time: ', time.time()-time_start)

    return unet

def trainUnet(cellpose_model_directory, training_images, num_epochs):

    #get the cellpose data from the images
    cpnet = resnet_torch.CPnet(nbase=[2,32,64,128,256],nout=3,sz=3)
    cpnet.load_model(cellpose_model_directory)
    cellpose_upsample, cellpose_output = [], []
    for image in training_images:
        upsample, output = get_pre_activations(image,cpnet)
        cellpose_upsample.append(upsample)
        cellpose_output.append(output)

    #augment the images
    images_torch = torch.from_numpy(np.array(training_images))
    images, cellpose_upsample, cellpose_output = augment_data(images_torch, cellpose_upsample, cellpose_output)

    train_dataset = ImageDataset(images_torch, cellpose_upsample, cellpose_output)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    #train the model
    unet = UNet(nbClasses=1)
    unet = unet.to('cuda:0')
    loss_fn = KDLoss()
    optimiser = torch.optim.SGD(unet.parameters(), lr=0.1, momentum=0.1)
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimiser, base_lr=0.0000001, max_lr=0.1)

    for epoch in range(num_epochs):
        unet = trainEpoch(unet, train_loader, loss_fn, optimiser, scheduler=scheduler, epoch_num=epoch)

    torch.save(unet.state_dict(), "C:\\Users\\rz200\\Documents\\development\\distillCellSegTrack\\pipeline\\models\\model3")
    #return the model

    return unet

def predict(testing_images, unet):
    predictions = []
    for image in testing_images:
        image = image.unsqueeze(1)
        pred = unet(image)