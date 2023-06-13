#importing the Cellpose model

from cellpose import models, core
from unet_architecture import UNet
import torch

#input: directory as string
#output: cellpose model
def get_cellpose_model(dir):
    #TODO: check if the directory is a folde or a file and dig into it if its a folder
    cellpose_model = models.CellposeModel(gpu=core.use_gpu(), pretrained_model=dir)
    return cellpose_model

def get_unet(dir):
    unet = UNet(encChannels=(1,32,64,128),decChannels=(128,64,32),nbClasses=1)
    unet.load_state_dict(torch.load(dir))
    return unet

#if __name__ == '__main__':
    #directory = "C:\\Users\\rz200\\Documents\\development\\distillCellSegTrack\\datasets\\Fluo-C2DL-Huh7\\01\\models\\CP_20230601_101328"
    #model = importModel(directory)
    #print (model)