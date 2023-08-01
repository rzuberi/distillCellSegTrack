# Distilling Cellpose (distillCellSegTrack)

# Objective

Here we are distilling the generalist cell segementation model [Cellpose](https://github.com/MouseLand/cellpose) into more efficient smaller models.

Pros of Cellpose:
- is moderately accurate in cell segmentations out of the box;
- has a human-in-the-loop feature to finetune models.

Cons of Cellpose:
- slow;
- memory intensive.

Student models have the same architecture as Cellpose (residual network) with less layers.
They are inherently faster and, by using less memory, we can segment multiple images using the same memory that Cellpose uses for only one image.

# How to distill a model

To distill a model in this project, there are three main steps: data gathering, model training, model evaluation.

1. Data gathering
The `data_utils.py` file has a `get_training_and_validation_loaders()` function which requires the path to the Cellpose model we wish to distill and the path to the images we wish to train our student model to segment. From there, we will have the the training data loader and validation data loader that we can use for the next step.

2. Model training
Train `train_utils.py` file has `train_model()` function which requires the following parameters in order:
- the architecture of the model, the smallest can be `[1,32]` and Cellpose usually has an architecture of `[1,32,64,128,256]`;
- the name of the model, as the model is trained its best version is saved locally;
- the training loader;
- the validation loader;
- optional inputs include the device we wish to train the model on (`cpu`, `cuda:0`, `mps`), if we wish to see the progress and manually setting the seed for the architecture of the model.

3. Model evaluation
There are two ways we evaluate the student model on test images against its teacher model Cellpose:
- the first is by binarising the test masks produced by both models and measuring the IoU;
- the second is by counting the number of individual cells or nuclei present in each mask.

# Data formation
The data we acquire from Cellpose is the following:
- the output of the before output layer in Cellpose which consists of 32 channels;
- the final 3-channel output consisting of flows in X, flows in Y and a probability map.

<img width="570" alt="[Screenshot 2023-08-01 at 13 05 58](https://www.nature.com/articles/s41592-020-01018-x
)" src="https://github.com/rzuberi/distillCellSegTrack/assets/56508673/de2b2b22-1432-46f9-916f-c7ab3d7e7e58">

To acquire this data, in our `data_utils.py` we stripped down some functions from Cellpose and modified them to make the architecture output this data.

# Model training
The loss used when training our model is a combination of the following:
- the 32-channel prediction minus the 32-channel ground-truth, meaned and squared;
- the X and Y flows are put through an MSE loss and summed with a BCE with logits loss of the probability map.

This loss is defined in KD_loss in `train_utils.py`.

