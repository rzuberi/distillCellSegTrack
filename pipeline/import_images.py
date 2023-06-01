#Importing the images to distill

import numpy as np
from os import listdir
from os.path import isfile, join
import tifffile
import matplotlib.pyplot as plt

#input: directory as string
#output: list of images as numpy arrays
def getImages(dir):
    onlyfiles = [f for f in listdir(dir) if isfile(join(dir, f))][1:]
    onlyfiles = [f for f in onlyfiles if f.endswith('.tif') or f.endswith('.tiff')]
    imgs = [np.squeeze(tifffile.imread(dir +  onlyfiles[i])) for i in range(len(onlyfiles))]
    return imgs

if __name__ == '__main__':
    directory = "C:\\Users\\rz200\\Documents\\development\\distillCellSegTrack\\pipeline\\uploads\\"
    imgs = getImages(directory)

    print(imgs[0].shape)