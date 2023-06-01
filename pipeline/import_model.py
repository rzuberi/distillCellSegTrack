#importing the Cellpose model

from cellpose import models, core


#input: directory as string
#output: cellpose model
def getModel(dir):
    #TODO: check if the directory is a folde or a file and dig into it if its a folder
    cellpose_model = models.CellposeModel(gpu=core.use_gpu(), pretrained_model=dir)
    return cellpose_model

if __name__ == '__main__':
    directory = "C:\\Users\\rz200\\Documents\\development\\distillCellSegTrack\\datasets\\Fluo-C2DL-Huh7\\01\\models\\CP_20230601_101328"
    model = importModel(directory)
    print (model)