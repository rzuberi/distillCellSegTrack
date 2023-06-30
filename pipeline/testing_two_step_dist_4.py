from unet_instance_2_chans import UNetModel as UNet_2chans

if __name__ == '__main__':

    model = UNet_2chans(encChannels=(2,32,64,128,256),decChannels=(256,128,64,32),nbClasses=3)
    model.load_weights("C:/Users/rz200/Documents/development/distillCellSegTrack/pipeline/unet_nuclei_hoechst_test_3")
    model.set_device('cuda')

    prediction = model.predict(combined_images[0])