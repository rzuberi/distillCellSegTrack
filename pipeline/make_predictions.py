
#these imports will get removed, they are just for testing
from import_images import getImages
#from import_model import getModel
import matplotlib.pyplot as plt
import torch
import numpy as np
import torch.nn.functional as F
from skimage import measure

def makePredictions(images, cellpose_model):
    probability_maps, cell_masks = [], []
    for image in images:
        print('a')
        masks, flows, styles = cellpose_model.eval(image, batch_size=1, channels=[0,0], diameter=cellpose_model.diam_labels)
        probability_maps.append(flows[2])
        cell_masks.append(masks)
    return probability_maps, cell_masks

def unet_segmentations(unet, images, destination_folder, file_names, device):
    segmentations = []
    #if destination_folder doesn't end with a slash or backslash, add one
    if destination_folder[-1] != '/' and destination_folder[-1] != '\\':
        destination_folder += '/'
    for (image, file_name) in zip(images,file_names):
        image = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
        if device != 'None':
            image = image.to(device)
        _, _, segmentation = unet(image)
        segmentation = F.sigmoid(segmentation)
        segmentation = segmentation.cpu().detach().numpy()[0][0]
        segmentation = np.where(segmentation > 0.5, 1.0, 0.0)
        segmentation = measure.label(segmentation, connectivity=2)
        print(type(segmentation))
        np.save(destination_folder + file_name + '_seg.npy', segmentation)
        

    return segmentations