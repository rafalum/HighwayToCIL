from torch import nn
from torchvision import models
import torch.nn.functional as F

from models.LinkNet import DecoderBlock

from functools import partial

### Code in part taken from https://github.com/wuhuikai/FastFCN ###

nonlinearity = partial(F.gelu)

class SeparableConv2d(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=1, dilation=1, bias=False, norm_layer=nn.BatchNorm2d):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size, stride, padding, dilation, groups=inplanes, bias=bias)
        self.bn = norm_layer(inplanes)
        self.pointwise = nn.Conv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x

class JPU(nn.Module):
    def __init__(self, width=512, norm_layer=nn.BatchNorm2d):
        super(JPU, self).__init__()
        
        self.dilate1 = nn.Sequential(SeparableConv2d(width, width, kernel_size=3, padding=1, dilation=1, bias=False),
                                       norm_layer(width),
                                       nn.ReLU(inplace=True))
        self.dilate2 = nn.Sequential(SeparableConv2d(width, width, kernel_size=3, padding=2, dilation=2, bias=False),
                                       norm_layer(width),
                                       nn.ReLU(inplace=True))
        self.dilate3 = nn.Sequential(SeparableConv2d(width, width, kernel_size=3, padding=4, dilation=4, bias=False),
                                       norm_layer(width),
                                       nn.ReLU(inplace=True))
        self.dilate4 = nn.Sequential(SeparableConv2d(width, width, kernel_size=3, padding=8, dilation=8, bias=False),
                                       norm_layer(width),
                                       nn.ReLU(inplace=True))
        
        self.dilate5 = nn.Sequential(SeparableConv2d(width, width, kernel_size=3, padding=16, dilation=16, bias=False),
                                       norm_layer(width),
                                       nn.ReLU(inplace=True))
        
    def forward(self, x):
        dilate1_out = self.dilate1(x)
        dilate2_out = self.dilate2(dilate1_out)
        dilate3_out = self.dilate3(dilate2_out)
        dilate4_out = self.dilate4(dilate3_out)
        dilate5_out = self.dilate5(dilate4_out)
        
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out + dilate5_out
        
        return out


class DLinkNet101_JPU(nn.Module):
    def __init__(self, num_classes=1):
        super(DLinkNet101_JPU, self).__init__()

        filters = [256, 512, 1024, 2048]
        resnet = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        
        self.dblock = JPU(2048)

        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        
        # Center
        e4 = self.dblock(e4)

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)
        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return F.sigmoid(out)