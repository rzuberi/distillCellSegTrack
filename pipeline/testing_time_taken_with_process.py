import numpy as np
import torch
from cellpose import resnet_torch
from cellpose import transforms, dynamics
import cv2
import time
from unet_architecture import UNet
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchmetrics.classification import BinaryJaccardIndex
import torchvision
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
from resnet_archi import CPnet
from cellpose import utils
import logging
from cellpose import models
core_logger = logging.getLogger(__name__)
tqdm_out = utils.TqdmToLogger(core_logger, level=logging.INFO)

def pred(x, network, return_conv=False, return_training_data=False,channels=None):
    """ convert imgs to torch and run network model and return numpy """
    #X = x.to('mps')
    X = x
    #self.net.eval()
    with torch.no_grad():
        if return_training_data == False:
            if channels == 1:
                X = X[:, 0, :, :]
                X = X.unsqueeze(1)
                #X = X.to('mps')
                y, style = network(X)
            else:
                #X = X.to('mps')
                y, style = network(X)
        else:
            channel_32_output, final_output = network(X, training_data=True)
            return channel_32_output, final_output
    del X
    y = y.to('mps')
    style = style.to('mps')
    if return_conv:
        conv = conv.to('mps')
        y = np.concatenate((y, conv), axis=1)
   
    return y, style
           

def run_tiled(imgi, network, augment=False, bsize=224, tile_overlap=0.1, return_conv=False, return_training_data=False,channels=None):
        """ run network in tiles of size [bsize x bsize]

        First image is split into overlapping tiles of size [bsize x bsize].
        If augment, tiles have 50% overlap and are flipped at overlaps.
        The average of the network output over tiles is returned.

        Parameters
        --------------

        imgi: array [nchan x Ly x Lx] or [Lz x nchan x Ly x Lx]

        augment: bool (optional, default False)
            tiles image with overlapping tiles and flips overlapped regions to augment

        bsize: int (optional, default 224)
            size of tiles to use in pixels [bsize x bsize]
         
        tile_overlap: float (optional, default 0.1)
            fraction of overlap of tiles when computing flows

        Returns
        ------------------

        yf: array [3 x Ly x Lx] or [Lz x 3 x Ly x Lx]
            yf is averaged over tiles
            yf[0] is Y flow; yf[1] is X flow; yf[2] is cell probability

        styles: array [64]
            1D array summarizing the style of the image, averaged over tiles

        """
        if not return_training_data:
            if imgi.ndim==4:
                batch_size = 8
                Lz, nchan = imgi.shape[:2]
                IMG, ysub, xsub, Ly, Lx = transforms.make_tiles(imgi[0], bsize=bsize,
                                                                augment=augment, tile_overlap=tile_overlap)
                ny, nx, nchan, ly, lx = IMG.shape
                batch_size *= max(4, (bsize**2 // (ly*lx))**0.5)
                yf = np.zeros((Lz, network.nout, imgi.shape[-2], imgi.shape[-1]), np.float32)
                styles = []
                if ny*nx > batch_size:
                    ziterator = trange(Lz, file=tqdm_out)
                    for i in ziterator:
                        yfi, stylei = run_tiled(imgi[i], augment=augment,
                                                    bsize=bsize, tile_overlap=tile_overlap)
                        yf[i] = yfi
                        styles.append(stylei)
                else:
                    # run multiple slices at the same time
                    ntiles = ny*nx
                    nimgs = max(2, int(np.round(batch_size / ntiles)))
                    niter = int(np.ceil(Lz/nimgs))
                    ziterator = trange(niter, file=tqdm_out)
                    for k in ziterator:
                        IMGa = np.zeros((ntiles*nimgs, nchan, ly, lx), np.float32)
                        for i in range(min(Lz-k*nimgs, nimgs)):
                            IMG, ysub, xsub, Ly, Lx = transforms.make_tiles(imgi[k*nimgs+i], bsize=bsize,
                                                                            augment=augment, tile_overlap=tile_overlap)
                            IMGa[i*ntiles:(i+1)*ntiles] = np.reshape(IMG, (ny*nx, nchan, ly, lx))
                        ya, stylea = pred(IMGa, network, channels=channels)
                        for i in range(min(Lz-k*nimgs, nimgs)):
                            y = ya[i*ntiles:(i+1)*ntiles]
                            if augment:
                                y = np.reshape(y, (ny, nx, 3, ly, lx))
                                y = transforms.unaugment_tiles(y, network)
                                y = np.reshape(y, (-1, 3, ly, lx))
                            yfi = transforms.average_tiles(y, ysub, xsub, Ly, Lx)
                            yfi = yfi[:,:imgi.shape[2],:imgi.shape[3]]
                            yf[k*nimgs+i] = yfi
                            stylei = stylea[i*ntiles:(i+1)*ntiles].sum(axis=0)
                            stylei /= (stylei**2).sum()**0.5
                            styles.append(stylei)
                return yf, np.array(styles)
            else:
                IMG, ysub, xsub, Ly, Lx = transforms.make_tiles(imgi, bsize=bsize,
                                                                augment=augment, tile_overlap=tile_overlap)
   
                ny, nx, nchan, ly, lx = IMG.shape
                IMG = np.reshape(IMG, (ny*nx, nchan, ly, lx))
                batch_size = 8
                niter = int(np.ceil(IMG.shape[0] / batch_size))
                nout = network.nout + 32*return_conv
                y = np.zeros((IMG.shape[0], nout, ly, lx))
                for k in range(niter):
                    irange = slice(batch_size*k, min(IMG.shape[0], batch_size*k+batch_size))
                    input_img = torch.from_numpy(IMG[irange])
                    y32, y0 = pred(input_img, network, return_training_data=False,channels=channels)
                    y0 = y0.cpu().detach().numpy()
                    y[irange] = y0.reshape(irange.stop-irange.start, y0.shape[-3], y0.shape[-2], y0.shape[-1])
                if augment:
                    y = np.reshape(y, (ny, nx, nout, bsize, bsize))
                    y = transforms.unaugment_tiles(y, network)
                    y = np.reshape(y, (-1, nout, bsize, bsize))
               
                yf = transforms.average_tiles(y, ysub, xsub, Ly, Lx)
                yf = yf[:,:imgi.shape[1],:imgi.shape[2]]
                return yf, y32
        else:
            tiles = []
            channel_32_outputs = []
            final_outputs = []
            IMG, ysub, xsub, Ly, Lx = transforms.make_tiles(imgi, bsize=bsize,
                                                            augment=augment, tile_overlap=tile_overlap)

            ny, nx, nchan, ly, lx = IMG.shape
            IMG = np.reshape(IMG, (ny*nx, nchan, ly, lx))
            batch_size = 8
            niter = int(np.ceil(IMG.shape[0] / batch_size))
            nout = network.nout + 32*return_conv
            y = np.zeros((IMG.shape[0], nout, ly, lx))
            for k in range(niter):
                irange = slice(batch_size*k, min(IMG.shape[0], batch_size*k+batch_size))
                input_img = torch.from_numpy(IMG[irange])

                channel_32_output, final_output = pred(input_img, network, return_training_data=True)

                tiles.append(input_img)
                channel_32_outputs.append(channel_32_output)
                final_outputs.append(final_output)

            return tiles, channel_32_outputs, final_outputs


def run_net(imgs, network, augment=False, tile=True, tile_overlap=0.1, bsize=224,
                 return_conv=False,return_training_data=False,channels=None):
        """ run network on image or stack of images

        (faster if augment is False)

        Parameters
        --------------

        imgs: array [Ly x Lx x nchan] or [Lz x Ly x Lx x nchan]

        rsz: float (optional, default 1.0)
            resize coefficient(s) for image

        augment: bool (optional, default False)
            tiles image with overlapping tiles and flips overlapped regions to augment

        tile: bool (optional, default True)
            tiles image to ensure GPU/CPU memory usage limited (recommended);
            cannot be turned off for 3D segmentation

        tile_overlap: float (optional, default 0.1)
            fraction of overlap of tiles when computing flows

        bsize: int (optional, default 224)
            size of tiles to use in pixels [bsize x bsize]

        Returns
        ------------------

        y: array [Ly x Lx x 3] or [Lz x Ly x Lx x 3]
            y[...,0] is Y flow; y[...,1] is X flow; y[...,2] is cell probability

        style: array [64]
            1D array summarizing the style of the image,
            if tiled it is averaged over tiles

        """  
        if imgs.ndim==4:  
            # make image Lz x nchan x Ly x Lx for net
            imgs = np.transpose(imgs, (0,3,1,2))
            detranspose = (0,2,3,1)
            return_conv = False
        else:
            # make image nchan x Ly x Lx for net
            imgs = np.transpose(imgs, (2,0,1))
            detranspose = (1,2,0)

        # pad image for net so Ly and Lx are divisible by 4
        imgs, ysub, xsub = transforms.pad_image_ND(imgs)
        # slices from padding
#         slc = [slice(0, self.nclasses) for n in range(imgs.ndim)] # changed from imgs.shape[n]+1 for first slice size
        slc = [slice(0, imgs.shape[n]+1) for n in range(imgs.ndim)]
        slc[-3] = slice(0, network.nout + 32*return_conv + 1)
        slc[-2] = slice(ysub[0], ysub[-1]+1)
        slc[-1] = slice(xsub[0], xsub[-1]+1)
        slc = tuple(slc)

        # run network
        if not return_training_data:
            if tile or augment or imgs.ndim==4:
                y, style = run_tiled(imgs, network=network, augment=augment, bsize=bsize,
                                        tile_overlap=tile_overlap,
                                        return_conv=return_conv,channels=channels)
            else:
                imgs = np.expand_dims(imgs, axis=0)
                y, style = network(imgs)
                y, style = y[0], style[0]
            style /= (style**2).sum()**0.5

            # slice out padding
            y = y[slc]
            # transpose so channels axis is last again
            y = np.transpose(y, detranspose)
           
            return y, style
        else:
            tiles, channel_32_outputs, final_outputs = run_tiled(imgs, network=network, augment=augment, bsize=bsize,
                                        tile_overlap=tile_overlap,
                                        return_conv=return_conv, return_training_data=return_training_data)
            return tiles, channel_32_outputs, final_outputs

def run_cp(x, network, compute_masks=True, normalize=True, invert=False,
            rescale=1.0, net_avg=False, resample=True,
            augment=False, tile=True, tile_overlap=0.1,
            cellprob_threshold=0.0,
            flow_threshold=0.4, min_size=15,
            interp=True, anisotropy=1.0, do_3D=False, stitch_threshold=0.0,
            return_training_data=False,channels=None):
   
    tic = time.time()
    shape = x.shape
    print('shape:',shape)
    nimg = shape[0]        
   
    bd, tr = None, None
    #tqdm_out = utils.TqdmToLogger(models_logger, level=logging.INFO)
    #iterator = trange(nimg,file=tqdm_out) if nimg>1 else range(nimg)
    iterator = range(nimg)
    styles = np.zeros((nimg, network.nbase[-1]), np.float32)
    if resample:
        dP = np.zeros((2, nimg, shape[1], shape[2]), np.float32)
        cellprob = np.zeros((nimg, shape[1], shape[2]), np.float32)
       
    else:
        dP = np.zeros((2, nimg, int(shape[1]*rescale), int(shape[2]*rescale)), np.float32)
        cellprob = np.zeros((nimg, int(shape[1]*rescale), int(shape[2]*rescale)), np.float32)

    if not return_training_data:
        for i in iterator:
            img = np.asarray(x[i])
            if normalize or invert:
                img = transforms.normalize_img(img, invert=invert)
            if rescale != 1.0:
                img = transforms.resize_image(img, rsz=rescale)
            #yf, style = self._run_nets(img, net_avg=net_avg,
            #                            augment=augment, tile=tile,
            #                            tile_overlap=tile_overlap)
            #print('normalised:',img)
            yf, style = run_net(img, network=network, augment=augment, tile=tile, tile_overlap=0.1, bsize=224,
                    return_conv=False,channels=channels)
            if resample:
                yf = transforms.resize_image(yf, shape[1], shape[2])

            cellprob[i] = yf[:,:,2]
            dP[:, i] = yf[:,:,:2].transpose((2,0,1))
            if network.nout == 4:
                if i==0:
                    bd = np.zeros_like(cellprob)
                bd[i] = yf[:,:,3]
        del yf, style
       
       
        net_time = time.time() - tic

        if compute_masks:
            tic=time.time()
            niter = 200 if (do_3D and not resample) else (1 / rescale * 200)
            masks, p = [], []
            resize = [shape[1], shape[2]] if not resample else None
            use_gpu = torch.cuda.is_available()
            device = torch.device('cuda' if use_gpu else 'cpu')
            for i in iterator:
                outputs = dynamics.compute_masks(dP[:,i], cellprob[i], niter=niter, cellprob_threshold=cellprob_threshold,
                                                        flow_threshold=flow_threshold, interp=interp, resize=resize,
                                                        use_gpu=use_gpu, device=device)
                masks.append(outputs[0])
                p.append(outputs[1])
               
            masks = np.array(masks)
            p = np.array(p)
           
            if stitch_threshold > 0 and nimg > 1:
                masks = utils.stitch3D(masks, stitch_threshold=stitch_threshold)
                masks = utils.fill_holes_and_remove_small_masks(masks, min_size=min_size)
       
            flow_time = time.time() - tic
            masks, dP, cellprob, p = masks.squeeze(), dP.squeeze(), cellprob.squeeze(), p.squeeze()
           
        else:
            masks, p = np.zeros(0), np.zeros(0)  #pass back zeros if not compute_masks
        return masks, styles, dP, cellprob, p
    else:
        tiled_images_input = []
        intermdiate_outputs = []
        flows_and_cellprob_output = []
        for i in iterator:
            img = np.asarray(x[i])
            if normalize or invert:
                img = transforms.normalize_img(img, invert=invert)
            if rescale != 1.0:
                img = transforms.resize_image(img, rsz=rescale)
            #yf, style = self._run_nets(img, net_avg=net_avg,
            #                            augment=augment, tile=tile,
            #                            tile_overlap=tile_overlap)

            #need to return tiles, 32_channel output and flow with cellprob
            tiles, channel_32_outputs, final_outputs = run_net(img, network=network, augment=augment, tile=tile, tile_overlap=0.1, bsize=224,
                    return_conv=False, return_training_data=True)
            tiled_images_input.append(tiles)
            intermdiate_outputs.append(channel_32_outputs)
            flows_and_cellprob_output.append(final_outputs)
        return tiled_images_input, intermdiate_outputs, flows_and_cellprob_output

def make_prediction(image, model, device, type):
    #type can be 'nuclei' or 'cell'

    x = image

    if not isinstance(x, list):
        x = np.array([x])
    #if there is only one channel, copy the channel to make it 2 channels
    if x.shape[0] == 1:
        x = np.repeat(x, 2, axis=0)
    if x.ndim < 4:
        x = x[np.newaxis,...]
    x = x.transpose((0,2,3,1))

    if type == 'nuclei':
        masks, styles, dP, cellprob, p = run_cp(x,model,channels=1,rescale=1.283732,return_training_data=False)
    elif type == 'cell':
        masks, styles, dP, cellprob, p = run_cp(x,model,channels=2,rescale=1.283732,return_training_data=False)
        masks = masks[0]

    return masks

def equalized_img(img):
    number_bins = 256
    image_histogram, bins = np.histogram(img.flatten(), number_bins, density=True)
    cdf = image_histogram.cumsum() # cumulative distribution function
    cdf = (number_bins-1) * cdf / cdf[-1] # normalize
    image_equalized = np.interp(img.flatten(), bins[:-1], cdf)
    image_equalized = image_equalized.reshape(img.shape)
    image_equalized = transforms.normalize_img(np.expand_dims(image_equalized,0))
    return image_equalized[0]

if __name__ == '__main__':

    #Load the test images
    image_folder = "/Users/rehanzuberi/Downloads/development/distillCellSegTrack/pipeline/saved_cell_images_1237"
    combined_images = []

    TYPE = 'nuclei'
    SEG = 'student'

    for filename in os.listdir(image_folder):
        if filename.endswith('.npy'):
            image_path = os.path.join(image_folder,filename)
            
            if TYPE == 'cell':
                numpy_image = np.load(image_path)
            elif TYPE == 'nuclei':
                numpy_image = np.load(image_path)[1]
                print(numpy_image.shape)
                numpy_image = equalized_img(numpy_image)
                print(numpy_image.shape)
                print('x')
            else:
                numpy_image = np.load(image_path)[1]
            combined_images.append(numpy_image)
    
    train_images, test_images = train_test_split(combined_images, test_size=0.2, random_state=42)    

    #make cellpose make a prediction on the test images
    if TYPE == 'cell':
        cellpose_model_directory = "/Users/rehanzuberi/Documents/Development/distillCellSegTrack/pipeline/CellPose_models/U2OS_Tub_Hoechst"
    elif TYPE == 'nuclei':
        cellpose_model_directory = "/Users/rehanzuberi/Documents/Development/distillCellSegTrack/pipeline/CellPose_models/Nuclei_Hoechst"
    
    if SEG == 'cellpose':
        segmentation_model = models.CellposeModel(gpu=True, model_type=cellpose_model_directory)
        masks = []
        for i in range(len(test_images)):
            mask, flows, styles = segmentation_model.eval(test_images[i])
            np.save('cellpose_nuclei_'+str(i)+'.npy', mask)
            masks.append(mask)
        #save the NPY images locally
        
    elif SEG == 'student':
        segmentation_model = CPnet(nbase=[1,32], nout=3, sz=3,
                        residual_on=True, style_on=True,
                        concatenation=False, mkldnn=False)
        segmentation_model.load_model('/Users/rehanzuberi/Downloads/Cellpose distilled models/resnet_nuc_32')
        masks = []
        for i in range(len(test_images)):
            mask = make_prediction(test_images[i], segmentation_model, 'mps', type=TYPE)
            np.save('resnet_32_nuclei_'+str(i)+'.npy', mask)
            masks.append(mask)


    #plot the images and the masks next to them
    if TYPE == 'cell':
        for i in range(len(test_images)):
            fig, ax = plt.subplots(1,2)
            ax[0].imshow(test_images[i][0])
            ax[1].imshow(masks[i][0])
            plt.show()
    elif TYPE == 'nuclei':
        for i in range(len(test_images)):
            fig, ax = plt.subplots(1,2)
            ax[0].imshow(test_images[i])
            ax[1].imshow(masks[i])
            plt.show()

    if 4 == 5:
        #Load the model

        #Make the inferences

        torch.manual_seed(1)
        model = CPnet(nbase=[1,32], nout=3, sz=3,
                        residual_on=True, style_on=True,
                        concatenation=False, mkldnn=False)
        start = time.time()
        for i in range(10):
            pred_mask = make_prediction(images[i], model, 'mps', 'nuclei')
        end = time.time() - start
        print(end)
        torch.cuda.empty_cache()

    