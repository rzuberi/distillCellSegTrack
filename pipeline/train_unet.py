#here we take the ground truth images and masks and train a unet and return it

from unet_architecture import UNet
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np

#these imports are for testing, they will be removed
from import_images import getImages
from import_model import getModel
from make_predictions import makePredictions
import time

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

def trainEpoch(unet, train_loader, loss_fn, optimiser, scheduler):
    time_start = time.time()
    
    unet.train()

    train_loss = 0
    for image, cellprob, cellmask in train_loader:
        (image,cellprob,cellmask) = (image.to('cuda:0'), cellprob.to('cuda:0'),cellmask.to('cuda:0')) # sending the data to the device (cpu or GPU)

        image = image.unsqueeze(1)
        pred = unet(image)# make a prediction
        
        cellprob = torch.unsqueeze(cellprob,1)
        cellmask = torch.unsqueeze(cellmask,1)

        #loss = loss_fn(pred, cellprob,10) # calculate the loss of that prediction
        loss = loss_fn(pred, cellprob, cellmask) # calculate the loss of that prediction
        train_loss += loss
    
        optimiser.zero_grad() # zero out the accumulated gradients
        loss.backward() # backpropagate the loss
        optimiser.step() # update model parameters
        scheduler.step()
    train_loss = train_loss.item()/len(train_loader)

    print('Training loss: ', train_loss, 'Time: ', time.time()-time_start)

    return unet

def split_image_set(set):
    splitted = []
    for image in set:
        for i in range(0, 1024, 256):
            for j in range(0, 1024, 256):
                sub_img = image[i:i+256, j:j+256]
                splitted.append(sub_img)
    return splitted

class KDLoss(torch.nn.Module):
    def __init__(self, alpha=0.5, beta = 0.5, temperature=2):
        super(KDLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature

    def forward(self, y_pred_logits, y_cp_true, y_cm_true):

        kd_loss = F.binary_cross_entropy_with_logits(y_pred_logits, y_cm_true)

        ce_loss = F.binary_cross_entropy_with_logits(y_pred_logits, y_cp_true)  # compute binary cross-entropy loss

        #y_pred_mask = torch.sigmoid(y_pred_logits)  # compute predicted probabilities
        #y_pred_mask = torch.where(y_pred_mask>0.4,1.0,0.0) # binarise
          # compute KL divergence loss
        
        loss = self.alpha * kd_loss + self.beta * ce_loss  # combine losses
        return loss

def trainUnet(training_images, training_probability_maps, training_cell_masks):
    #normalise the probability maps
    pm_normalised = [(image-np.min(image))/(np.max(image)-np.min(image)) for image in training_probability_maps]
    #binarise the cell masks
    cm_binary = [np.where(image > 0, 1.0, 0.0) for image in training_cell_masks]

    #pad the images, probability maps and cell masks to 1024x1024
    images_pad = [np.pad(img,((0,1024-img.shape[0]),(0,1024-img.shape[1])),mode='constant',constant_values=0) for img in training_images]
    pm_norm_pad = [np.pad(img,((0,1024-img.shape[0]),(0,1024-img.shape[1])),mode='constant',constant_values=0) for img in pm_normalised]
    cm_bin_pad = [np.pad(img,((0,1024-img.shape[0]),(0,1024-img.shape[1])),mode='constant',constant_values=0) for img in cm_binary]

    #split the images, probability maps and cell masks into 256x256 images
    images_pad_split = split_image_set(images_pad)
    pm_norm_pad_split = split_image_set(pm_norm_pad)
    cm_bin_pad_split = split_image_set(cm_bin_pad)

    #remove the images from cm_bin_pad_split that only have 0s, and remove them from images_pad_split and pm_norm_pad_split
    images_pad_split_filtered = [images_pad_split[i] for i in range(len(cm_bin_pad_split)) if np.sum(cm_bin_pad_split[i]) > 0]
    pm_norm_pad_split_filtered = [pm_norm_pad_split[i] for i in range(len(cm_bin_pad_split)) if np.sum(cm_bin_pad_split[i]) > 0]
    cm_bin_pad_split_filtered = [img for img in cm_bin_pad_split if np.sum(img) > 0]

    print(len(images_pad_split_filtered), len(pm_norm_pad_split_filtered), len(cm_bin_pad_split_filtered))

    images_torch = [torch.from_numpy(np.array(images_pad_split_filtered[i])) for i in range(len(images_pad_split_filtered))]
    pm_torch = [torch.from_numpy(np.array(pm_norm_pad_split_filtered[i])) for i in range(len(pm_norm_pad_split_filtered))]
    cm_torch = [torch.from_numpy(np.array(cm_bin_pad_split_filtered[i])) for i in range(len(cm_bin_pad_split_filtered))]

    train_dataset = ImageDataset(images_torch, pm_torch, cm_torch)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    unet = UNet()
    unet = unet.to('cuda:0')
    loss_fn = KDLoss()
    optimiser = torch.optim.SGD(unet.parameters(), lr=0.1, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimiser, base_lr=0.001, max_lr=0.1)
    num_epochs = 10
    
    for epoch in range(num_epochs):
        unet = trainEpoch(unet, train_loader, loss_fn, optimiser, scheduler)

    return unet

def getIoU(pred, target):
    intersection = np.logical_and(target, pred)
    union = np.logical_or(target, pred)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score

if __name__ == '__main__':
    images_directory = "C:\\Users\\rz200\\Documents\\development\\distillCellSegTrack\\pipeline\\uploads\\"
    images = getImages(images_directory)

    directory = "C:\\Users\\rz200\\Documents\\development\\distillCellSegTrack\\datasets\\Fluo-C2DL-Huh7\\01\\models\\CP_20230601_101328"
    cellpose_model = getModel(directory)

    training_images = images[:5]
    training_probability_maps, training_cell_masks = makePredictions(images[:1], cellpose_model)
    testing_images = images[5:10]
    testing_probability_maps, testing_cell_masks = makePredictions(images[5:10], cellpose_model)

    unet = trainUnet(training_images, training_probability_maps, training_cell_masks)

    print('IoU :',getIoU(unet(testing_images[0]), testing_cell_masks[0]))