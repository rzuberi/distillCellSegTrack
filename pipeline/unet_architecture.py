import torch
from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import ReLU
from torch.nn import ModuleList
from torch.nn import MaxPool2d
from torch.nn import ConvTranspose2d
from torch.nn import BatchNorm2d
from torchvision.transforms import CenterCrop

class Block(Module):

    def __init__(self, inChannels, outChannels):
        super().__init__()
        self.conv1 = Conv2d(inChannels, outChannels, 3, padding=1)
        self.relu = ReLU()
        self.conv2 = Conv2d(outChannels, outChannels, 3, padding=1)
        self.batchnorm = BatchNorm2d(outChannels, eps=1e-5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        
        #adding more
        #x = self.relu(x)
        #x = self.conv2(x)
        #x = self.relu(x)
        #x = self.conv2(x)
        #x = self.relu(x)
        #x = self.conv2(x)

        return x #CONV->RELU->CONV

class Encoder(Module):

    def __init__(self, channels=(1, 16, 32, 64)):
        super().__init__()
        # store the encoder blocks and maxpooling layer
        self.encBlocks = ModuleList([Block(channels[i], channels[i + 1])
                                    for i in range(len(channels) - 1)])
        self.pool = MaxPool2d(2)

    def forward(self, x):
		# initialize an empty list to store the intermediate outputs
        blockOutputs = []
        # loop through the encoder blocks
        for block in self.encBlocks:
            # pass the inputs through the current encoder block, store
            # the outputs, and then apply maxpooling on the output
            x = block(x)
            blockOutputs.append(x)
            x = self.pool(x)
        # return the list containing the intermediate outputs
        return blockOutputs

class Decoder(Module):
    def __init__(self, channels=(64, 32, 16)):
        super().__init__()
        # initialize the number of channels, upsampler blocks, and
        # decoder blocks
        self.channels = channels
        self.upconvs = ModuleList([ConvTranspose2d(channels[i], channels[i + 1], 2, 2)for i in range(len(channels) - 1)])
        self.dec_blocks = ModuleList([Block(channels[i], channels[i + 1]) for i in range(len(channels) - 1)])

    def forward(self, x, encFeatures):
    # loop through the number of channels
        for i in range(len(self.channels) - 1):
            # pass the inputs through the upsampler blocks
            x = self.upconvs[i](x)
            # crop the current features from the encoder blocks,
            # concatenate them with the current upsampled features,
            # and pass the concatenated output through the current
            # decoder block
            encFeat = self.crop(encFeatures[i], x)
            x = torch.cat([x, encFeat], dim=1)
            x = self.dec_blocks[i](x)
            # return the final decoder output
        return x

    def crop(self, encFeatures, x):
        # grab the dimensions of the inputs, and crop the encoder
        # features to match the dimensions
        (_, _, H, W) = x.shape
        encFeatures = CenterCrop([H, W])(encFeatures)
        # return the cropped features
        return encFeatures

def batchconv(in_channels, out_channels, sz):
    return torch.nn.Sequential(
        torch.nn.BatchNorm2d(in_channels, eps=1e-5),
        torch.nn.ReLU(inplace=True),
        torch.nn.Conv2d(in_channels, out_channels, sz, padding=sz//2),
    )  

class UNet(Module):
    def __init__(self, encChannels=(1, 16, 32, 64), decChannels=(64, 32), nbClasses=1, retainDim=True, outSize=(128, 128)):
        super().__init__()
        # initialize the encoder and decoder
        self.encoder = Encoder(encChannels)
        self.decoder = Decoder(decChannels)
        # initialize the regression head and store the class variables
        #self.head = Conv2d(decChannels[-1], nbClasses, 1)
        self.head = batchconv(in_channels=32, out_channels=nbClasses, sz=1)
        self.retainDim = retainDim
        self.outSize = outSize

    def forward(self, x):
        x = x.type(torch.float32)
        encFeatures = self.encoder(x)
        decFeatures = self.decoder(encFeatures[::-1][0], encFeatures[::-1][1:])
        map = self.head(decFeatures)
        #map = map.repeat(1, 3, 1, 1)
        return encFeatures, decFeatures, map