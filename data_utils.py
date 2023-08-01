import os
import numpy as np
import torch
import scipy
from cellpose import transforms, models
from resnet_archi import CPnet
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

def pred(x, network):
    """ convert imgs to torch and run network model and return numpy """
    X = x
    with torch.no_grad():
        channel_32_output, final_output = network(X, training_data=True)
        return channel_32_output, final_output

def make_tiles_and_reshape(img, bsize, augment, tile_overlap):
    tiles, _, _, _, _ = transforms.make_tiles(img, bsize=bsize, augment=augment, tile_overlap=tile_overlap)
    ny, nx, nchan, ly, lx = tiles.shape
    return np.reshape(tiles, (ny*nx, nchan, ly, lx))

def run_tiled(imgi, network, unnormed, augment=False, bsize=224, tile_overlap=0.1, ):
    tiles = []
    channel_32_outputs = []
    final_outputs = []

    IMG = make_tiles_and_reshape(imgi, bsize, augment, tile_overlap)

    IMG_flatfield_corrected = flatfield_correction(unnormed)
    IMG_flatfield_corrected = make_tiles_and_reshape(IMG_flatfield_corrected, bsize, augment, tile_overlap)

    batch_size = 8
    niter = int(np.ceil(IMG.shape[0] / batch_size))

    for k in range(niter):
        irange = slice(batch_size * k, min(IMG.shape[0], batch_size * (k + 1)))
        input_img = torch.from_numpy(IMG[irange])

        channel_32_output, final_output = pred(input_img, network)
        channel_32_outputs.append(channel_32_output)
        final_outputs.append(final_output)

        irange = slice(batch_size * k, min(IMG_flatfield_corrected.shape[0], batch_size * (k + 1)))
        input_img = torch.from_numpy(IMG_flatfield_corrected[irange])
        tiles.append(input_img)

    return tiles, channel_32_outputs, final_outputs

def run_net(imgs, network, unnormed, augment=False, tile_overlap=0.1, bsize=224,
                 return_conv=False,return_training_data=False):

        # make image nchan x Ly x Lx for net
        imgs = np.transpose(imgs, (2,0,1))
        unnormed = np.transpose(unnormed, (2,0,1))

        # pad image for net so Ly and Lx are divisible by 4
        imgs, _, _ = transforms.pad_image_ND(imgs)
        unnormed, _, _ = transforms.pad_image_ND(unnormed)

        tiles, channel_32_outputs, final_outputs = run_tiled(imgs, network=network, augment=augment, bsize=bsize,
                                    tile_overlap=tile_overlap,unnormed=unnormed)
        return tiles, channel_32_outputs, final_outputs

def run_cp(x, network, normalize=True, invert=False, rescale=1.0, augment=False, tile=True):
   
    iterator = range(x.shape[0])
        
    tiled_images_input = []
    intermdiate_outputs = []
    flows_and_cellprob_output = []
    for i in iterator:
        img = np.asarray(x[i])
        img_unnormed = img.copy()
        if normalize or invert:
            img = transforms.normalize_img(img, invert=invert)
        if rescale != 1.0:
            img = transforms.resize_image(img, rsz=rescale)
            img_unnormed = transforms.resize_image(img_unnormed, rsz=rescale)
        
        tiles, channel_32_outputs, final_outputs = run_net(img, network=network, augment=augment, tile_overlap=0.1, bsize=224,
                return_conv=False, return_training_data=True, unnormed=img_unnormed)
        tiled_images_input.append(tiles)
        intermdiate_outputs.append(channel_32_outputs)
        flows_and_cellprob_output.append(final_outputs)
    return tiled_images_input, intermdiate_outputs, flows_and_cellprob_output

import torch

def get_clean_data(data_input):
    clean_data = []

    for i in range(len(data_input[0])):
        # Remove the channel at dimension 1
        #tiles = np.delete(data_input[0][i], 1, 1)
        tiles = np.squeeze(data_input[0][i])
        
        for tile in tiles:
            clean_data.append(tile)

    return torch.stack(clean_data)

def get_clean_train_data(tiled_images_input, intermediate_outputs, flows_and_cellprob_output):
    tiled_images_input_train_clean = get_clean_data(tiled_images_input)
    tiled_intermediate_outputs_train_clean = get_clean_data(intermediate_outputs)
    tiled_flows_and_cellprob_output_train_clean = get_clean_data(flows_and_cellprob_output)

    tiled_images_input_train_clean = tiled_images_input_train_clean.reshape(-1, 2, 224, 224)

    return tiled_images_input_train_clean, tiled_intermediate_outputs_train_clean, tiled_flows_and_cellprob_output_train_clean

def get_data_cp_clean(unet,combined_images,rescale):
    tiled_images_final = []
    intermediate_outputs_final = []
    flows_and_cellprob_output_final = []

    for i in range(len(combined_images)):

        x = combined_images[i]

        if not isinstance(x, np.ndarray):
            x = np.array(x)

        while x.ndim < 4:
            x = x[np.newaxis,...]

        x = x.transpose((0,2,3,1))

        tiled_images_input, intermdiate_outputs, flows_and_cellprob_output = run_cp(x,unet,rescale=rescale)
        tiled_images_input_train_clean, tiled_intermediate_outputs_train_clean, tiled_flows_and_cellprob_output_train_clean = get_clean_train_data(tiled_images_input, intermdiate_outputs, flows_and_cellprob_output)
        tiled_images_final.append(tiled_images_input_train_clean)
        intermediate_outputs_final.append(tiled_intermediate_outputs_train_clean)
        flows_and_cellprob_output_final.append(tiled_flows_and_cellprob_output_train_clean)
    
    tiled_images_final = torch.cat(tiled_images_final, dim=0)
    intermediate_outputs_final = torch.cat(intermediate_outputs_final, dim=0)
    flows_and_cellprob_output_final =  torch.cat(flows_and_cellprob_output_final, dim=0)

    return tiled_images_final, intermediate_outputs_final, flows_and_cellprob_output_final

def gaussian_filter(pixel_data, sigma):
    """Perform gaussian filter with the given radius"""
    def fn(image):
        return scipy.ndimage.gaussian_filter(
            image, sigma, mode="constant", cval=0
        )
    mask = np.ones(pixel_data.shape)
    bleed_over = fn(mask)
    smoothed_image = fn(pixel_data)
    output_image = smoothed_image / (bleed_over + np.finfo(float).eps)
    return output_image

def flatfield_correction(image):
    if len(image.shape) == 2:
        image = np.array([image]) 

    sigma = 30

    if image.shape[0] > 2:
        corrected_per_channel = []
        for channel in image:
            blurred_img = gaussian_filter(channel, sigma)
            blurred_img_meaned = blurred_img / blurred_img.mean()
            result = channel / blurred_img_meaned
            corrected_per_channel.append(result)
        final_img = np.array(corrected_per_channel)
    else:
        blurred_img = gaussian_filter(image, sigma)
        blurred_img_meaned = blurred_img / blurred_img.mean()
        final_img = image / blurred_img_meaned
    return final_img

def get_training_and_validation_loaders(cellpose_model_directory, image_folder, channel=None):
    # Whole cellpose model
    segmentation_model = models.CellposeModel(gpu=True, model_type=cellpose_model_directory)
    rescale = segmentation_model.diam_mean / segmentation_model.diam_labels

    # Only the architecture, easier to extract the data
    cpnet = CPnet(nbase=[2, 32, 64, 128, 256], nout=3, sz=3, residual_on=True)
    cpnet.load_model(cellpose_model_directory)

    combined_images = []
    for filename in os.listdir(image_folder):
        if filename.endswith('.npy'):
            image_path = os.path.join(image_folder, filename)
            numpy_image = np.load(image_path)
            combined_images.append(numpy_image)
    
    if channel != None:
        combined_images = np.array(combined_images)
        combined_images = np.delete(combined_images, channel, 1)
        combined_images = np.repeat(combined_images, 2, axis=1)
    else:
        combined_images = np.array(combined_images)

    combined_images = combined_images[27:28]

    tiled_images_final, intermediate_outputs_final, flows_and_cellprob_output_final = get_data_cp_clean(cpnet, combined_images, rescale=rescale)

    train_images_tiled, val_images_tiled, train_upsamples, val_upsamples, train_ys, val_ys = train_test_split(
        tiled_images_final, intermediate_outputs_final, flows_and_cellprob_output_final,
        test_size=0.1, random_state=42)

    if channel != None:
        #remove the second channel fo train_images_tiled and val_images_tiled
        train_images_tiled = np.delete(train_images_tiled, channel, 1)
        val_images_tiled = np.delete(val_images_tiled, channel, 1)

    train_dataset = ImageDataset(train_images_tiled, train_upsamples, train_ys)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    validation_dataset = ImageDataset(val_images_tiled, val_upsamples, val_ys)
    validation_loader = DataLoader(validation_dataset, batch_size=8, shuffle=True)

    return train_loader, validation_loader

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

if __name__ == '__main__':
    cellpose_model_directory = "/Users/rehanzuberi/Documents/Development/distillCellSegTrack/pipeline/CellPose_models/U2OS_Tub_Hoechst"
    image_folder = "/Users/rehanzuberi/Downloads/development/distillCellSegTrack/pipeline/saved_cell_images_1237"
    
    train_loader, validation_loader = get_training_and_validation_loaders(cellpose_model_directory, image_folder, channel = 0)
