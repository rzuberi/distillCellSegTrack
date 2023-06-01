
#these imports will get removed, they are just for testing
from import_images import getImages
from import_model import getModel
import matplotlib.pyplot as plt

def makePredictions(images, cellpose_model):
    probability_maps, cell_masks = [], []
    for image in images:
        print('a')
        masks, flows, styles = cellpose_model.eval(image, batch_size=1, channels=[0,0], diameter=cellpose_model.diam_labels)
        probability_maps.append(flows[2])
        cell_masks.append(masks)
    return probability_maps, cell_masks
