from import_images import getImages
#from import_model import getModel
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
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import torchvision.transforms as T

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
    downsample_return = downsample[1].squeeze(0)
    downsample_return = downsample_return.cpu().detach().numpy().tolist()

    for (k, image) in enumerate(downsample_return):
        downsample_return[k] = cv2.resize(np.array(image), dsize=(512, 512), interpolation=cv2.INTER_NEAREST)
    downsample_return = np.array(downsample_return)
    downsample_return = torch.from_numpy(downsample_return)


    style = cpnet.make_style(downsample[-1])
    upsample = cpnet.upsample(style, downsample, cpnet.mkldnn)

    output = cpnet.output(upsample)
    output = output.squeeze(0)
    output = output[2]
    
    print(output.shape)
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

    return downsample_return, upsample, output

class ImageDataset(Dataset):
    def __init__(self, image, upsample, cellprob):
        self.image = image
        self.upsample = upsample
        self.cellprob = cellprob

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        img = self.image[idx]
        upsample = self.upsample[idx]
        cellprob = self.cellprob[idx]
        return img, upsample, cellprob
    
def get_data(images_torch, cp_upsamples, cp_outputs):

    train_images, test_images, train_upsamples, test_upsamples, train_cellprob, test_cellprob = train_test_split(images_torch, cp_upsamples, cp_outputs, test_size=0.1, random_state=42)

    images_torch_rotated = []
    for image in train_images:
        images_torch_rotated.append(image)
        for i in range(3):
            image = torch.rot90(image, 1, [0, 1])
            images_torch_rotated.append(image)

    images_torch_rotated_flipped = []
    for image in images_torch_rotated:
        images_torch_rotated_flipped.append(image)
        images_torch_rotated_flipped.append(torch.flip(image, [1]))

    cp_upsamples_rotated = []
    for upsample in train_upsamples:
        cp_upsamples_rotated.append(upsample)
        for i in range(3):
            upsample = torch.rot90(upsample, 1, [1, 2])
            cp_upsamples_rotated.append(upsample)

    cp_upsamples_rotated_flipped = []
    for upsample in cp_upsamples_rotated:
        cp_upsamples_rotated_flipped.append(upsample)
        cp_upsamples_rotated_flipped.append(torch.flip(upsample, [2]))

    cp_outputs_rotated = []
    for output in train_cellprob:
        cp_outputs_rotated.append(output)
        for i in range(3):
            output = torch.rot90(output, 1, [0, 1])
            cp_outputs_rotated.append(output)

    cp_outputs_rotated_flipped = []
    for output in cp_outputs_rotated:
        cp_outputs_rotated_flipped.append(output)
        cp_outputs_rotated_flipped.append(torch.flip(output, [1]))
    
    return images_torch_rotated_flipped, cp_upsamples_rotated_flipped, cp_outputs_rotated_flipped, test_images, test_upsamples, test_cellprob

class LF_one(torch.nn.Module):
    def __init__(self):
        super(LF_one, self).__init__()
    def forward(self, y_32_pred, y_32_true):
        y_32_pred = F.sigmoid(y_32_pred)
        y_32_true = F.sigmoid(y_32_true)
        y_32_loss = F.mse_loss(y_32_pred, y_32_true.float())
        return y_32_loss
    
class LF_two(torch.nn.Module):
    def __init__(self):
        super(LF_two, self).__init__()
    def forward(self, y_3_pred, y_3_true):
        #transform = T.Resize(1024)
        #y_3_true = transform(y_3_true)

        y_3_pred = F.sigmoid(y_3_pred)
        y_3_true = F.sigmoid(y_3_true)
        
        y_3_loss = F.binary_cross_entropy(y_3_pred, y_3_true.float())
    
        y_3_pred = y_3_pred.view(-1)
        y_3_true = y_3_true.view(-1)
        intersection = (y_3_pred * y_3_true).sum()
        total = (y_3_pred + y_3_true).sum()
        union = total - intersection 
        IoU_loss = 1 - (intersection + 1)/(union + 1)

        return y_3_loss * 0.5 + IoU_loss * 0.5

def trainEpoch(unet, train_loader, test_loader, loss_fn, optimiser, scheduler, epoch_num, step, output_directory, model_name):
    time_start = time.time()
    
    unet.train()

    train_loss = 0
    for image, upsample, cp_output in train_loader:
        (image, upsample, cp_output) = (image.to('cuda:0'),upsample.to('cuda:0'),cp_output.to('cuda:0')) # sending the data to the device (cpu or GPU)

        image = image.unsqueeze(1)
        y_16_pred, y_32_pred, y_3_pred = unet(image)
        y_32_pred = y_32_pred.squeeze(1)
        y_3_pred = y_3_pred.squeeze(1)
                
        if step == 1:
            transform = T.Resize(512)
            y_32_pred = transform(y_32_pred)
            loss = loss_fn(y_32_pred,  upsample)
        elif step == 2:
            transform = T.Resize(512)
            y_3_pred = transform(y_3_pred)
            loss = loss_fn(y_3_pred, cp_output)

        train_loss += loss
        optimiser.zero_grad() # zero out the accumulated gradients
        loss.backward() # backpropagate the loss
        optimiser.step() # update model parameters
        if scheduler is not None:
            scheduler.step()
    train_loss = train_loss.item()/len(train_loader)

    if step == 2:
        iou_score = 0
        for image, upsample, cp_output in test_loader:
            (image, upsample, cp_output) = (image.to('cuda:0'),upsample.to('cuda:0'),cp_output.to('cuda:0'))

            image = image.unsqueeze(1)
            y_16_pred, y_32_pred, y_3_pred = unet(image)
            
            y_32_pred = y_32_pred.squeeze(1)
            y_3_pred = y_3_pred.squeeze(1)
            
            y_3_pred = F.sigmoid(y_3_pred)
            y_3_pred = y_3_pred.cpu().detach().numpy().tolist()
            y_3_pred = np.array(y_3_pred)
            y_3_pred = np.squeeze(y_3_pred)
            y_3_pred = y_3_pred > 0.5
            y_3_pred = y_3_pred.astype(int)

            transform = T.Resize(1024)
            cp_output = transform(cp_output)

            cp_output = cp_output.cpu().detach().numpy().tolist()
            cp_output = np.array(cp_output)
            cp_output = np.squeeze(cp_output)
            cp_output = cp_output > 0.5
            cp_output = cp_output.astype(int)

            intersection = np.logical_and(cp_output, y_3_pred)
            union = np.logical_or(cp_output, y_3_pred)
            iou_score += np.sum(intersection) / np.sum(union)

        for image, upsample, cp_output in test_loader:
            
            (image, cp_output) = (image.to('cuda:0'),cp_output.to('cuda:0'))

            image = image.unsqueeze(1)
            y_16_pred, y_32_pred, y_3_pred = unet(image)
            y_32_pred = y_32_pred.squeeze(1)
            y_3_pred = y_3_pred.squeeze(1)
            
            y_3_pred = F.sigmoid(y_3_pred)
            y_3_pred = y_3_pred.cpu().detach().numpy().tolist()
            y_3_pred = np.array(y_3_pred)
            y_3_pred = np.squeeze(y_3_pred)
            y_3_pred = y_3_pred > 0.5
            y_3_pred = y_3_pred.astype(int)

            cp_output = cp_output.cpu().detach().numpy().tolist()
            cp_output = np.array(cp_output)
            cp_output = np.squeeze(cp_output)
            cp_output = cp_output > 0.5
            cp_output = cp_output.astype(int)
            
            if epoch_num % 10 == 0:
                plt.subplot(1,4,1)
                plt.imshow(y_3_pred)
                plt.subplot(1,4,2)
                plt.imshow(y_3_pred)
                plt.subplot(1,4,3)
                plt.imshow(cp_output)
                plt.subplot(1,4,4,)
                plt.imshow(cp_output)
                plt.show()
            
            break


        iou_score = iou_score/len(test_loader)

    if epoch_num is None:
        print('Training loss: ', train_loss, 'Time: ', time.time()-time_start)
    else:
        if step == 1:
            print('Epoch ', epoch_num, 'Training loss: ', train_loss, 'Time: ', time.time()-time_start)
        elif step == 2:
            print('Epoch ', epoch_num, 'Training loss: ', train_loss, 'IoU score: ', iou_score, 'Time: ', time.time()-time_start)
            if epoch_num % 5 == 0 and epoch_num != 0:
                torch.save(unet.state_dict(), output_directory + model_name + "_epoch_" + str(epoch_num) + "_IoU_" + str(round(iou_score*10)))

    return unet

#function to get iou between two torch tensors
def get_iou(y_pred, y_true):
    y_pred = F.sigmoid(y_pred)
    y_pred = y_pred.cpu().detach().numpy().tolist()
    y_pred = np.array(y_pred)
    y_pred = np.squeeze(y_pred)
    y_pred = y_pred > 0.5
    y_pred = y_pred.astype(int)

    y_true = F.sigmoid(y_true)
    y_true = y_true.cpu().detach().numpy().tolist()
    y_true = np.array(y_true)
    y_true = np.squeeze(y_true)
    y_true = y_true > 0.5
    y_true = y_true.astype(int)

    intersection = np.logical_and(y_true, y_pred)
    union = np.logical_or(y_true, y_pred)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score

def train_unet(cellpose_model_directory, images_directory, output_directory, n_epochs, model_name):
    #if output_directory doesn't end with a slash or backslash, add one
    if output_directory[-1] != '/' and output_directory[-1] != '\\':
        output_directory += '/'

    cpnet = resnet_torch.CPnet(nbase=[2,32,64,128,256],nout=3,sz=3)
    cpnet.load_model(cellpose_model_directory)

    #images_directory = "C:\\Users\\rz200\\Documents\\development\\distillCellSegTrack\\pipeline\\uploads\\"
    file_names, images = getImages(images_directory)
    images_torch = torch.from_numpy(np.array(images))[:2]

    cp_upsamples = []
    cp_outputs = []
    for image in images[:2]:
        downsample, upsample, output = get_pre_activations(image,cpnet)
        cp_upsamples.append(upsample)
        cp_outputs.append(output)

    train_images, train_upsamples, train_maps, test_images, test_upsamples, test_maps = get_data(images_torch, cp_upsamples, cp_outputs)

    train_dataset = ImageDataset(train_images, train_upsamples, train_maps)
    test_dataset = ImageDataset(test_images, test_upsamples, test_maps)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    #Train model
    unet = UNet(encChannels=(1,32,64,128),decChannels=(128,64,32),nbClasses=1)
    unet = unet.to('cuda:0')

    optimiser = torch.optim.SGD(unet.parameters(), lr=1, momentum=0.1)

    scheduler = torch.optim.lr_scheduler.CyclicLR(optimiser, base_lr=0.0001, max_lr=1)
    loss_fn = LF_one()
    for epoch in range(n_epochs):
        unet = trainEpoch(unet, train_loader, test_loader, loss_fn, optimiser, scheduler=scheduler, epoch_num=epoch, step=1, output_directory=output_directory, model_name=model_name)

    scheduler = torch.optim.lr_scheduler.CyclicLR(optimiser, base_lr=0.0001, max_lr=0.01)
    loss_fn = LF_two()
    for epoch in range(n_epochs):
        unet = trainEpoch(unet, train_loader, test_loader, loss_fn, optimiser, scheduler=scheduler, epoch_num=epoch, step=2, output_directory=output_directory, model_name=model_name)

    torch.save(unet.state_dict(), output_directory + model_name + "_final")

    #test U-Net
    iou_score = 0
    for image, upsample, cp_output in test_loader:
        (image, upsample, cp_output) = (image.to('cuda:0'),upsample.to('cuda:0'),cp_output.to('cuda:0'))

        image = image.unsqueeze(1)
        y_16_pred, y_32_pred, y_3_pred = unet(image)

        transform = T.Resize(1024)
        cp_output = transform(cp_output)
        iou_score += get_iou(y_3_pred, cp_output)

    iou_score = iou_score/len(test_loader)

    return unet, iou_score