from data_utils import get_training_and_validation_loaders
from train_utils import train_model
from method_seeding import find_seed

if __name__ == '__main__':

    cellpose_model_directory = "/Users/rehanzuberi/Downloads/development/distillCellSegTrack/cellpose_models/Nuclei_Hoechst"
    image_folder = "/Users/rehanzuberi/Downloads/development/distillCellSegTrack/saved_cell_images_1237"
    device = 'mps'
    
    #channel 0: nuclei
    #channel 1: cell
    #no channel: both cell and nuclei
    train_loader, validation_loader = get_training_and_validation_loaders(cellpose_model_directory, image_folder, channel = 0, augment = False)

    n_base = [1,32] #model's dimensions
    num_iter = 10 #number of seeds to test for seed searching
    epochs_per_model = 1 #number of epochs to train the models that we test with different seeds
    seed = find_seed(n_base=n_base, epochs=1, train_loader=train_loader, validation_loader=validation_loader, num_iter=num_iter, device=device,progress=True)

    student_model = train_model([1,32],100,'student_models/resnet_testing',train_loader, validation_loader, device=device,progress=True,seed=seed)

