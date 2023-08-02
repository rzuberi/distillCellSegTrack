from data_utils import get_training_and_validation_loaders
from train_utils import train_model

if __name__ == '__main__':

    cellpose_model_directory = "/Users/rehanzuberi/Downloads/development/distillCellSegTrack/cellpose_models/Nuclei_Hoechst"
    image_folder = "/Users/rehanzuberi/Downloads/development/distillCellSegTrack/saved_cell_images_1237"
    
    train_loader, validation_loader = get_training_and_validation_loaders(cellpose_model_directory, image_folder, channel = 0, augment = True)

    #TODO: include here the "find best seed" method

    student_model = train_model([1,32],100,'student_models/resnet_testing',train_loader, validation_loader, device='mps',progress=True,seed=23944)

