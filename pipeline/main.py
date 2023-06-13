#here I'll just define the functions that will be called

#as a rule, only imports from other files in the pipeline folder are allowed
from import_images import getImages
from import_model import get_cellpose_model, get_unet
from make_predictions import makePredictions, unet_segmentations
from train_model_clean import train_unet

if __name__ == '__main__':

    #1
    #get the images that have to be distilled
    images_directory = "C:\\Users\\rz200\\Documents\\development\\distillCellSegTrack\\pipeline\\uploads\\"
    file_names, images = getImages(images_directory)
    print(file_names)

    #2
    #get the trained Cellpose model
    cellpose_model_directory = "C:\\Users\\rz200\\Documents\\development\\distillCellSegTrack\\datasets\\Fluo-C2DL-Huh7\\01\\models\\CP_20230601_101328"
    cellpose_model = get_cellpose_model(cellpose_model_directory)

    #3
    #use the Cellpose model to make 5 predictions from the images
    #training_images = images[:5]
    #training_probability_maps, training_cell_masks = makePredictions(images[:5], cellpose_model)
    #testing_images = images[5:10]
    #testing_probability_maps, testing_cell_masks = makePredictions(images[5:10], cellpose_model)

    #4
    #use these images to train the U-Net
    unet, iou_score = train_unet(cellpose_model_directory, images_directory, n_epochs=2, model_name='unet_1')
    print(iou_score)

    #import model
    unet = get_unet("C:\\Users\\rz200\\Documents\\development\\distillCellSegTrack\\pipeline\\models\\unet_1")

    #5
    #use the U-Net to make all the other predictions
    #return the images with their original names + '_seg.npy'
    unet_segmentations(unet, images, images_directory, file_names)