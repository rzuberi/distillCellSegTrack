#here we take the ground truth images and masks and train a unet and return it

from unet_architecture import UNet
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torchvision
import numpy as np
# import train test split
from sklearn.model_selection import train_test_split

#these imports are for testing, they will be removed
from import_images import getImages
#from import_model import getModel
#from make_predictions import makePredictions
import time
import matplotlib.pyplot as plt
#import faulthandler
#faulthandler.enable()

class ImageDataset(Dataset):
    def __init__(self, image, cellmask):
        self.image = image
        self.cellmask = cellmask

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        img = self.image[idx]
        cellmask = self.cellmask[idx]
        return img, cellmask

def trainEpoch(unet, train_loader, loss_fn, optimiser, scheduler):
    time_start = time.time()
    
    unet.train()

    train_loss = 0
    train_iou = 0
    for image, cellmask in train_loader:
        image, cellmask = image.type(torch.float32), cellmask.type(torch.float32)
        (image,cellmask) = (image.to('mps'),cellmask.to('mps')) # sending the data to the device (cpu or GPU)

        image = image.unsqueeze(1)
        pred = unet(image)# make a prediction
        
        cellmask = torch.unsqueeze(cellmask,1)

        #loss = loss_fn(pred, cellprob,10) # calculate the loss of that prediction
        loss = loss_fn(pred, cellmask) # calculate the loss of that prediction
        #loss = loss_fn(pred, cellmask )
        train_loss += loss

        pred = torch.sigmoid(pred)
        pred = torch.where(pred > 0.5, 1.0, 0.0)
        iou = getIoU(cellmask,pred)
        train_iou += iou

        
    
        optimiser.zero_grad() # zero out the accumulated gradients
        loss.backward() # backpropagate the loss
        optimiser.step() # update model parameters
        scheduler.step()
    train_loss = train_loss/len(train_loader)
    train_iou = train_iou/len(train_loader)

    print('Training loss: ', train_loss, 'Time: ', time.time()-time_start, 'iou:', train_iou )

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
    def __init__(self, alpha=1.0, beta = 0.5, temperature=2):
        super(KDLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature

    def forward(self, y_pred_logits, y_cm_true):

        kd_loss = F.binary_cross_entropy_with_logits(y_pred_logits, y_cm_true)
        
        #binarised = torch.where(F.sigmoid(y_pred_logits) > 0.4, 1.0, 0.0)
        #ce_loss = F.binary_cross_entropy(binarised, y_cp_true)  # compute binary cross-entropy loss
        #loss = F.binary_cross_entropy_with_logits(y_pred_logits, y_cm_true)
        #y_pred_mask = torch.sigmoid(y_pred_logits)  # compute predicted probabilities
        #y_pred_mask = torch.where(y_pred_mask>0.4,1.0,0.0) # binarise
          # compute KL divergence loss
        
        loss = self.alpha * kd_loss # combine losses
        return loss
        
        #print('CE LOSS', ce_loss.item(), 'KD LOSS', kd_loss.item())

class BCEandIoU(torch.nn.Module):
    def __init__(self, alpha=1.0, beta = 1.0):
        super(BCEandIoU, self).__init__()
        self.alpha = alpha
        self.beta = beta
    
    def forward(self, y_pred, y_true):
        bce = F.binary_cross_entropy_with_logits(y_pred, y_true)
        bce = bce*self.alpha

        y_pred = torch.where(torch.sigmoid(y_pred) > 0.5, 1.0, 0)
        iou_loss = 1 - (getIoU(y_pred,y_true))
        iou_loss = iou_loss*self.beta

        return bce + iou_loss


class DiceLoss(torch.nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice
    

def trainUnet(training_images, training_cell_masks):
    #normalise the probability maps
    #pm_normalised = [(image-np.min(image))/(np.max(image)-np.min(image)) for image in training_probability_maps]
    #binarise the cell masks
    cm_binary = [np.where(image > 0, 1.0, 0.0) for image in training_cell_masks]

    #pad the images, probability maps and cell masks to 1024x1024
    images_pad = [np.pad(img,((0,1024-img.shape[0]),(0,1024-img.shape[1])),mode='constant',constant_values=0) for img in training_images]
    cm_bin_pad = [np.pad(img,((0,1024-img.shape[0]),(0,1024-img.shape[1])),mode='constant',constant_values=0) for img in cm_binary]

    #split the images, probability maps and cell masks into 256x256 images
    images_pad_split = split_image_set(images_pad)
    cm_bin_pad_split = split_image_set(cm_bin_pad)

    #augment with rotations
    images_rotated = []
    cm_rotated = []
    for i in range(len(images_pad_split)):
        for j in range(1,4):
            images_rotated.append(np.rot90(images_pad_split[i],j))
            cm_rotated.append(np.rot90(cm_bin_pad_split[i],j))

    #remove the images from cm_bin_pad_split that only have 0s, and remove them from images_pad_split and pm_norm_pad_split
    images_pad_split_filtered = [images_rotated[i] for i in range(len(cm_rotated)) if np.sum(cm_rotated[i]) > 0]
    cm_bin_pad_split_filtered = [img for img in cm_rotated if np.sum(img) > 0]

    print(len(images_pad_split_filtered), len(cm_bin_pad_split_filtered))

    images_torch = [torch.from_numpy(np.array(images_pad_split_filtered[i])) for i in range(len(images_pad_split_filtered))]
    cm_torch = [torch.from_numpy(np.array(cm_bin_pad_split_filtered[i])) for i in range(len(cm_bin_pad_split_filtered))]

    train_dataset = ImageDataset(images_torch, cm_torch)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    unet = UNet()
    unet = unet.to('mps')
    #loss_fn = KDLoss()
    loss_fn = BCEandIoU()
    #loss_fn = IoULoss()
   # loss_fn = torchvision.ops.distance_box_iou_loss()
    #loss_fn = DiceLoss()
    optimiser = torch.optim.SGD(unet.parameters(), lr=0.01, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimiser, base_lr=0.0001, max_lr=0.01)
    num_epochs = 20
    
    for epoch in range(num_epochs):
        unet = trainEpoch(unet, train_loader, loss_fn, optimiser, scheduler)

    return unet

def getIoU(pred, target):
    intersection = torch.logical_and(target, pred)
    union = torch.logical_or(target, pred)
    iou_score = torch.sum(intersection) / torch.sum(union)
    return iou_score

def split_image(img):
    # Split the input array into smaller arrays of size 256x256
    sub_images = []
    for i in range(0, img.shape[0], 256):
        for j in range(0, img.shape[1], 256):
            sub_img = img[i:i+256, j:j+256]
            sub_images.append(sub_img)
    return sub_images

def combine_images(sub_images):
    # Create a NumPy array of size 1024x1024 to store the combined image
    img = np.zeros((1024, 1024))
    # Combine the smaller arrays into the larger image
    k = 0
    for i in range(0, img.shape[0], 256):
        for j in range(0, img.shape[1], 256):
            img[i:i+256, j:j+256] = sub_images[k]
            k += 1     
    return img

if __name__ == '__main__':

    #Get the cell images
    images_directory = "/Users/rehanzuberi/Documents/Development/distillCellSegTrack/datasets/Fluo-C2DL-Huh7/01/"
    images = getImages(images_directory)
    images = [images[0],images[2],images[6],images[14],images[15],images[19],images[25],images[27]]
    print(len(images))

    #Get the masks
    masks_directory = "/Users/rehanzuberi/Documents/Development/distillCellSegTrack/datasets/Fluo-C2DL-Huh7/01_GT/SEG/"
    cell_masks = getImages(masks_directory)
    print(len(cell_masks))

    X_train, X_test, y_train, y_test = train_test_split(images, cell_masks, test_size=0.1)
    print(len(X_train),len(X_test),len(y_train),len(y_test))

    unet = trainUnet(X_train, y_train)

    print('DONE TRAINING')

    #split the testing image
    test_image_split = split_image(X_test[0])
    test_mask_preds = []
    for image in test_image_split:
        pred = unet(torch.from_numpy(image).unsqueeze(0).unsqueeze(0).to('mps'))
        pred = torch.sigmoid(pred)
        pred = torch.where(pred > 0.5, 1.0, 0.0)
        test_mask_preds.append(pred.cpu().detach().numpy())
    full_pred = combine_images(test_mask_preds)
    plt.subplot(1,2,1)
    plt.imshow(y_test[0])
    plt.subplot(1,2,2)
    plt.imshow(full_pred)
    plt.show()
    print('iou:',getIoU(y_test[0],full_pred))

    #THE PREDICTION NEEDS TO BE MADE ON THE 256x256 images not full size

    testing_images = images[5:10]
    testing_probability_maps, testing_cell_masks = makePredictions(images[5:10], cellpose_model)

    #pad the testing images
    #split them into 256x256
    #get a prediction for each
    #get the IoU metric for each and average it

    pred = unet(torch.from_numpy(training_images[0]).unsqueeze(0).unsqueeze(0).to('mps'))
    pred = pred.squeeze(0).squeeze(0).cpu().detach().numpy()
    #sigmoid pred
    pred_bin = 1/(1+np.exp(-pred))
    print('unique',np.unique(pred_bin))
    pred_bin = np.where(pred_bin > 0.5, 1.0, 0.0)

    print(getIoU(pred_bin, training_cell_masks[0]))

    #the important note for now is that the training loss is not good enough for now and that there needs to be some post-processing of the images to close the holes in the masks and also we have not yet added the "instance mask" creation part which is one line anyway