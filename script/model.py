
import torch
import torch.nn as nn
import torch.nn.functional as F 

class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, bn=True):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2,1, bias=False)]
        if bn: layers.append(nn.BatchNorm2d(out_size, momentum=0.8))
        layers.append(nn.LeakyReLU(0.2))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        retur self.model(x)

class UNetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4,2,1,bias = False),
            nn.BatchNorm2d(out_size, momentum=0.8),
            nn.ReLU(inplace=True)
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x,skip_input),1)
        return x


class GeneratorFunieGAN(nn.Module):

    def __init__(self,inchannels = 3, out_channles=3):
        super(GeneratorFunieGAN, self).__init__()
        self.down1 = UNetDown(inchannels, 32, bn=False)
        self.down2 = UNetDown(32, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 256)
        self.down5 = UNetDown(256,256, bn = False)

        self.up1 = UNetUp(256,256)
        self.up2 = UNetUp(512,256)
        self.up3 = UNetUp(512, 128)
        self.up4 = UNetUp(256, 32)
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d(1,0,1,0),
            nn.Conv2d(64, out_channles, 4, padding = 1),
            nn.Tanh(),
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        u1 = self.up1(d5,d4)
        u2 = self.up2(u1,d3)
        u3 = self.up3(u2,d2)
        u4 = self.up4(u3,d1)
        return self.final(u4)

class DiscriminatorFunieGAN(nn.Module)
    def __init__(self, in_channels=3):
        super(DiscriminatorFunieGAN,self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if bn: layers.append(nn.BatchNorm2d(out_filters, momentum=0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels*2, 32, bn=False),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
            *discriminator_block(128,256),
            nn.ZeroPad2d(1,0,1,0),
            nn.Conv2d(256, 1, 4, padding=1, bias = False)
        )

    def forward(self, img_A, img_B):
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)


def Weights_Normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)

class VGG19_PercepLoss(nn.Module):
    def __init__(self, _pertrained_=True):
        super(VGG19_PercepLoss,self).__init__()
        self.vgg = models.vgg19(pertrained=_pertrained_).features
        for param in self.vgg.parameters():
            param.requires_grad_(False)
    
    def get_features(self, image, layers=None):
        if layers is None:
            layers = {'30':'conv5_2'}
        features = {}
        x = image
        for name,layers in self.vgg._modules.items():
            x = layers(x)
            if name in layers:
                features[layers[name]] = x 
        return features

    def forward(self, pred, true, layer='conv5_2'):
        true_f = self.get_features(true)
        pred_f = self.get_features(pred)
        return torch.mean((true_f[layer]-pred_f[layer])**2)
        
    