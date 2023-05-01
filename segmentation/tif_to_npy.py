import numpy as np
from os import listdir
from os.path import isfile, join
import tifffile


if __name__ == '__main__':

    #import the masks in 01_GT
    path = 'datasets/Fluo-N2DL-HeLa/'
    num_imgs = 92


    masks_path = path + '01_GT/TRA/'
    onlyfiles = [f for f in listdir(masks_path) if isfile(join(masks_path, f))][1:]
    onlyfiles.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    if num_imgs > len(onlyfiles): num_imgs = len(onlyfiles)
    masks = [np.squeeze(tifffile.imread(masks_path +  onlyfiles[i])) for i in range(num_imgs)]
    print(type(masks[0]))

    #for i in range(len(masks)):
        #the num has to be a number with 3 digits that starts with 000 then 001
    #    num = str(i).zfill(3)
    #    print(num)
    #    name_of_file = 't' + num + '_seg'
    #    print(name_of_file)
    #    np.save(name_of_file, masks[i])