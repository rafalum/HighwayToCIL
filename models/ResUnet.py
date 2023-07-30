import torch
from torch import nn
from torchvision import models

### ResUnet with a four layer encoder ###

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        shortcut = self.shortcut(x)

        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv2(x)

        x += shortcut

        return x

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(ResBlock, self).__init__()

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride[0], padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride[1], padding=1)

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride[0], padding=0),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        shortcut = self.shortcut(x)

        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv1(x)

        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv2(x)

        x += shortcut

        return x

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        resnet = models.resnet34(pretrained=True)

        self.conv = ConvBlock(3, 64)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward(self, x):
        x = self.conv(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return [x1, x2, x3, x4]

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.up0 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.up1 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.up3 = nn.ConvTranspose2d(128, 64, 2, 2)

        self.res_block0 = ResBlock(1024, 512, [(1, 1), (1, 1)])
        self.res_block1 = ResBlock(512, 256, [(1, 1), (1, 1)])
        self.res_block2 = ResBlock(256, 128, [(1, 1), (1, 1)])
        self.res_block3 = ResBlock(128, 64, [(1, 1), (1, 1)])

    def forward(self, x, from_encoder):

        x = self.up0(x)
        x = torch.cat((x, from_encoder[3]), dim=1)
        x = self.res_block0(x)
        
        x = self.up1(x)
        x = torch.cat((x, from_encoder[2]), dim=1)
        x = self.res_block1(x)

        x = self.up2(x)
        x = torch.cat((x, from_encoder[1]), dim=1)
        x = self.res_block2(x)

        x = self.up3(x)
        x = torch.cat((x, from_encoder[0]), dim=1)
        x = self.res_block3(x)

        return x

class ResUNet(nn.Module):
    def __init__(self):
        super(ResUNet, self).__init__()

        self.encoder = Encoder()
        self.bridge = ResBlock(512, 1024, [(2, 2), (1, 1)])
        self.decoder = Decoder()
        self.head = nn.Sequential(nn.Conv2d(64, 1, 1), 
                                  nn.Sigmoid())

    def forward(self, x):

        from_encoder = self.encoder(x)
        bridge = self.bridge(from_encoder[3])
        x = self.decoder(bridge, from_encoder)

        return self.head(x)