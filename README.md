# distillCellSegTrack (distilling Cellpose)

# Objective

Here we are distilling the generalist cell segementation model Cellpose into more efficient smaller models.

Pros of Cellpose:
- is moderately accurate in cell segmentations out of the box;
- has a human-in-the-loop feature to finetune models.

Cons of Cellpose:
- slow;
- memory intensive.

Student models have the same architecture as Cellpose (residual network) with less layers.
They are inherently faster and, by using less memory, we can segment multiple images using the same memory that Cellpose uses for only one image.

# How to distill a model
- first get data with data_utils
- then train model with train_utils
- thirdly evaluate the model with eval_utils
- use the images within saved_cell_images_1237

# Data formation
- getting the data from Cellpose

# Model training
- the loss is a combination from the 32 channel and last 3 channels

# Evaluation
- IoU and cell counting

