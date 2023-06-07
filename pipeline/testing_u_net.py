from unet_architecture import UNet
import torch

from cellpose import resnet_torch

from resnet_architecture import CPnet_s


from ptflops import get_model_complexity_info

if __name__ == "__main__":
    unet = UNet()
    x = torch.randn(1, 1, 1024, 1024)

    macs, params = get_model_complexity_info(unet, (1, 1024, 1024), as_strings=True,
                                           print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    
    directory = "C:\\Users\\rz200\\Documents\\development\\distillCellSegTrack\\datasets\\Fluo-C2DL-Huh7\\01\\models\\CP_20230601_101328"
    cpnet = resnet_torch.CPnet(nbase=[2,32,64,128,256],nout=3,sz=3)
    cpnet.load_model(directory)

    macs, params = get_model_complexity_info(cpnet, (2, 1024, 1024), as_strings=True,
                                           print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    cpnet = CPnet_s(nbase=[2,32,64,128,256],nout=3,sz=3)
    cpnet.load_model(directory)
    
    macs, params = get_model_complexity_info(cpnet, (2, 1024, 1024), as_strings=True,
                                           print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))