#here I'll just define the functions that will be called

#as a rule, only imports from other files in the pipeline folder are allowed
from import_images import getImages
from import_model import getModel
from make_predictions import makePredictions
from train_model import trainUnet

if __name__ == '__main__':

    #1
    #get the images that have to be distilled
    images_directory = "C:\\Users\\rz200\\Documents\\development\\distillCellSegTrack\\pipeline\\uploads\\"
    images = getImages(images_directory)

    #2
    #get the trained Cellpose model
    cellpose_model_directory = "C:\\Users\\rz200\\Documents\\development\\distillCellSegTrack\\datasets\\Fluo-C2DL-Huh7\\01\\models\\CP_20230601_101328"
    cellpose_model = getModel(cellpose_model_directory)

    #3
    #use the Cellpose model to make 5 predictions from the images
    training_images = images[:5]
    training_probability_maps, training_cell_masks = makePredictions(images[:5], cellpose_model)
    testing_images = images[5:10]
    testing_probability_maps, testing_cell_masks = makePredictions(images[5:10], cellpose_model)

    #4
    #use these images to train the U-Net
    unet = trainUnet(cellpose_model_directory, training_images, 400)

    #5
    #use the U-Net to make all the other predictions
    #return the images with their original names + '_seg.npy'