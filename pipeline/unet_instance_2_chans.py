import numpy as np
import torch
from unet_architecture import UNet
import torch
import numpy as np
import cv2
from scipy.ndimage import label
from cellpose import transforms, dynamics
import matplotlib.pyplot as plt

class UNetModel:
    def __init__(self, encChannels=(2,32,64,128,256),decChannels=(256,128,64,32),nbClasses=3):
        self.encChannels = encChannels
        self.decChannels = decChannels
        self.nbClasses = nbClasses
        self.device = torch.device('cpu')  # Default device is CPU
        self.model = None

    def set_device(self, device):
        if device == 'cpu':
            self.device = torch.device('cpu')
        elif device == 'cuda' and torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            print('CUDA is not available. Using CPU.')
            self.device = torch.device('cpu')

    def load_weights(self, weights_path):
        self.model = UNet(self.encChannels, self.decChannels, self.nbClasses)
        self.model.load_state_dict(torch.load(weights_path))
        self.model.to(self.device)
        self.model.eval()

    def predict(self, images, normalize=True):
        if isinstance(images, np.ndarray):
            images = [images]  # Convert single image to list

        predictions = []
        for image in images:
            if image.shape[0] != 2:
                image = np.array([image,image]) #2-channel images needed for model input
            instance_segmentation, outputs, dP = self._process_image(image,normalize)
            predictions.append([instance_segmentation,outputs, dP])

        return predictions

    def _process_image(self, image, normalize=True):

        # Pre-processing image (normalisation and tiling)
        if normalize:
            image = transforms.normalize_img(image)

        tiles, ysub, xsub, Ly, Lx = transforms.make_tiles(image, bsize=224, 
                                                augment=True, tile_overlap=0.1)

        # Predicting
        ny, nx, nchan, ly, lx = tiles.shape
        tiles = np.reshape(tiles, (ny*nx, nchan, ly, lx))
        batch_size = 1
        niter = int(np.ceil(tiles.shape[0] / batch_size))
        nout = 3 + 32*False
        y_unet = np.zeros((tiles.shape[0], nout, ly, lx))


        for k in range(niter):
            irange = np.arange(batch_size*k, min(tiles.shape[0], batch_size*k+batch_size))
            #print('IMG irange',tiles[irange].shape)
            _, _, y0_unet = self.model(torch.from_numpy(tiles[irange]))
            y0_unet = y0_unet.cpu().detach().numpy()
            y_unet[irange] = y0_unet.reshape(len(irange), y0_unet.shape[-3], y0_unet.shape[-2], y0_unet.shape[-1])

        yf = transforms.average_tiles(y_unet, ysub, xsub, Ly, Lx)
        yf = yf[:,:image.shape[1],:image.shape[2]]

        shape = image.shape
        dP = np.zeros((2, 1, int(shape[1]*1), int(shape[2]*1)), np.float32)
        cellprob = np.zeros((1, int(shape[1]*1), int(shape[2]*1)), np.float32)

        yf_t = yf.transpose(1,2,0)
        cellprob[0] = yf_t[:,:,2]
        cellprob = cellprob[0]
        dP[:, 0] = yf[:,:,:2]

        # Post processing
        reassembled_image = 1 / (1 + np.exp(-cellprob))
        reassembled_image = reassembled_image > 0.5
        instance_segmentation = self._binary_to_instance(reassembled_image)

        outputs = dynamics.compute_masks(yf_t[:,:,:2].transpose((2,0,1)), yf_t[:,:,2], niter=156, cellprob_threshold=0,
                                                         flow_threshold=0.4, interp=True, resize=None, 
                                                         use_gpu=True, device="cuda:0")
        print('ggggg')
        return instance_segmentation, outputs, dP

    def _binary_to_instance(self, img):
        image = img.astype(np.uint8)
        image = cv2.morphologyEx(image, cv2.MORPH_OPEN, np.ones((1, 1), dtype=int))

        a = image
        img = image

        border = cv2.dilate(img, None, iterations=1)
        border = border - cv2.erode(border, None)

        dt = cv2.distanceTransform(img, 2, 3)
        dt = ((dt - dt.min()) / (dt.max() - dt.min()) * 255).astype(np.uint8)
        _, dt = cv2.threshold(dt, 130, 255, cv2.THRESH_BINARY)
        lbl, ncc = label(dt)
        lbl = lbl * (255 / (ncc + 1))
        # Completing the markers now.
        lbl[border == 255] = 255

        lbl = lbl.astype(np.int32)
        a = cv2.cvtColor(a, cv2.COLOR_GRAY2RGB)
        cv2.watershed(a, lbl)

        lbl[lbl == -1] = 0
        lbl = lbl.astype(np.uint8)

        result = 255 - lbl

        # Find the most frequent value in the array and set it to zero
        unique_values, counts = np.unique(result, return_counts=True)
        sorted_indices = np.argsort(counts)
        most_frequent_values = unique_values[sorted_indices[-2:]]
        result[np.isin(result, most_frequent_values)] = 0

        unique_values = np.unique(result)  # Find the unique values in the array
        sorted_values = np.sort(unique_values)  # Sort the unique values
        value_index_map = {value: index for index, value in enumerate(sorted_values)}
        ordered_array = np.vectorize(lambda x: value_index_map[x])(result)

        return ordered_array
