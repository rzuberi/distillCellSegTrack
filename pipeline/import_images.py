#Importing the images to distill

import numpy as np
from os import listdir
from os.path import isfile, join
import tifffile
import matplotlib.pyplot as plt

#input: directory as string
#output: list of images as numpy arrays
def getImages(dir):
    onlyfiles = [f for f in listdir(dir) if isfile(join(dir, f))]
    onlyfiles = [f for f in onlyfiles if f.endswith('.tif') or f.endswith('.tiff')]
    file_names = [f.split('.')[0] for f in onlyfiles]
    #if the last character in dir is not a slash or backslash, add one
    if dir[-1] != '/' and dir[-1] != '\\':
        dir += '/'
    imgs = [np.squeeze(tifffile.imread(dir +  onlyfiles[i])) for i in range(len(onlyfiles))]
    return file_names, imgs

if __name__ == '__main__':
    #directory = "C:/Users/rz200/Documents/development/distillCellSegTrack/pipeline/uploads/"
    #imgs = getImages(directory)

    directory = "C:/Users/rz200/Documents/development/distillCellSegTrack/test_segs/"
    onlyfiles = [f for f in listdir(directory) if isfile(join(directory, f))]
    onlyfiles = [f for f in onlyfiles if f.endswith('.npy')]
    for file in onlyfiles:
        seg = np.load(directory + file)
        plt.imshow(seg)
        plt.show()