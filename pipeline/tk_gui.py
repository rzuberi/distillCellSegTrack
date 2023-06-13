import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
import numpy as np
from functools import partial

from import_images import getImages
from import_model import get_cellpose_model, get_unet
from make_predictions import makePredictions, unet_segmentations
from train_model_clean import train_unet

# Function to handle button click event
def execute_code(images_entry, model_entry, output_entry, device, train_model):
    images_directory = images_entry.get()
    cellpose_model_directory = model_entry.get()
    output_directory = output_entry.get()

    file_names, images = getImages(images_directory)
    print(file_names)

    #cellpose_model = get_cellpose_model(model_directory)

    # makePredictions(images, cellpose_model)  # Uncomment this line when ready to use
    
    #TODO: check if the images are normalised before training the model
    #TODO: save model in the output directory
    print(cellpose_model_directory, images_directory)


    if train_model:
        unet, iou_score = train_unet(cellpose_model_directory, images_directory, output_directory, n_epochs=40, model_name='unet_1_saved')
        print(iou_score)
    else:
        unet = get_unet(cellpose_model_directory)
    
    if device != 'None':
        unet = unet.to(device)
    #unet = get_unet(model_directory)
    unet_segmentations(unet, images, output_directory, file_names, device)

# Function to handle browse button click event
def browse_directory(entry):
    directory = filedialog.askdirectory()
    entry.delete(0, tk.END)
    entry.insert(tk.END, directory)

# Create the GUI window
window = tk.Tk()
window.title("Image Processing")
window.geometry("400x310")

# Create labels and entry fields for input directories
tk.Label(window, text="Images Directory:").pack()
images_entry = tk.Entry(window)
images_entry.pack()
default_images_directory = "C:/Users/rz200/Documents/development/distillCellSegTrack/pipeline/uploads"
images_entry.insert(tk.END, default_images_directory)
browse_images_btn = tk.Button(window, text="Browse", command=partial(browse_directory, images_entry))
browse_images_btn.pack()

tk.Label(window, text="Model Directory:").pack()
model_entry = tk.Entry(window)
model_entry.pack()
default_model_directory = "C:/Users/rz200/Documents/development/distillCellSegTrack/datasets/Fluo-C2DL-Huh7/01/models/CP_20230601_101328"
model_entry.insert(tk.END, default_model_directory)
def browse_file(entry):
    file_path = filedialog.askopenfilename()
    entry.delete(0, tk.END)
    entry.insert(tk.END, file_path)
browse_model_btn = tk.Button(window, text="Browse", command=partial(browse_file, model_entry))
browse_model_btn.pack()

tk.Label(window, text="Output Directory:").pack()
output_entry = tk.Entry(window)
output_entry.pack()
default_output_directory = "C:/Users/rz200/Documents/development/distillCellSegTrack/test_segs"
output_entry.insert(tk.END, default_output_directory)
browse_output_btn = tk.Button(window, text="Browse", command=partial(browse_directory, output_entry))
browse_output_btn.pack()

tk.Label(window, text="Device:").pack()
device_text = tk.Text(window, height=1, width=10)
device_text.pack()
default_device = "cuda:0"
device_text.insert(tk.END, default_device)


# Create a variable to store the state of the "train model" checkbox
train_model_var = tk.BooleanVar()

def get_train_model_state():
    train_model_state = train_model_var.get()
    # Use the train_model_state variable in your code
    print("Train Model State:", train_model_state)
    
# Create the checkbox
train_model_checkbox = tk.Checkbutton(window, text="Train Model", variable=train_model_var)
train_model_checkbox.pack()

# Create button to execute the code
execute_btn = tk.Button(window, text="Execute Code", command=partial(execute_code, images_entry, model_entry, output_entry, device_text.get("1.0", "end-1c"), train_model=train_model_var.get()))
execute_btn.pack()

# Start the GUI event loop
window.mainloop()
