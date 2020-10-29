import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import models
from torch.utils.data import Dataset
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms as transforms
from PIL import Image
import yaml
import numpy as np 

def Weights_Normal(m):
    # initialize weights as Normal(mean, std)
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)

class VGG19_PercepLoss(nn.Module):
    """ Calculates perceptual loss in vgg19 space
    """
    def __init__(self, _pretrained_=True):
        super(VGG19_PercepLoss, self).__init__()
        self.vgg = models.vgg19(pretrained=_pretrained_).features
        for param in self.vgg.parameters():
            param.requires_grad_(False)

    def get_features(self, image, layers=None):
        if layers is None: 
            layers = {'30': 'conv5_2'} # may add other layers
        features = {}
        x = image
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in layers:
                features[layers[name]] = x
        return features

    def forward(self, pred, true, layer='conv5_2'):
        true_f = self.get_features(true)
        pred_f = self.get_features(pred)
        return torch.mean((true_f[layer]-pred_f[layer])**2)

class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, bn=True):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if bn: layers.append(nn.BatchNorm2d(out_size, momentum=0.8))
        layers.append(nn.LeakyReLU(0.2))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_size, momentum=0.8),
            nn.ReLU(inplace=True),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)
        return x


class GenNN(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(GenNN, self).__init__()
        # encoding layers
        self.down1 = UNetDown(in_channels, 32, bn=False)
        self.down2 = UNetDown(32, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 256)
        self.down5 = UNetDown(256, 256, bn=False)
        # decoding layers
        self.up1 = UNetUp(256, 256)
        self.up2 = UNetUp(512, 256)
        self.up3 = UNetUp(512, 128)
        self.up4 = UNetUp(256, 32)
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(64, out_channels, 4, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        u1 = self.up1(d5, d4)
        u2 = self.up2(u1, d3)
        u3 = self.up3(u2, d2)
        u4 = self.up4(u3, d1)
        return self.final(u4)

class DisNN(nn.Module):
    """ A 4-layer Markovian discriminator as described in the paper
    """
    def __init__(self, in_channels=3):
        super(DisNN, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            #Returns downsampling layers of each discriminator block
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if bn: layers.append(nn.BatchNorm2d(out_filters, momentum=0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels*2, 32, bn=False),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(256, 1, 4, padding=1, bias=False)
        )

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)

class Loaddata(Dataset):
    def __init__(self,cfg):
        transforms_ = [
            transforms.Resize((cfg["img_height"], cfg["img_width"]), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
        self.transform = transforms.Compose(transforms_)
        self.len = cfg["data_size"]

    def __getitem__(self, index):
        unloader = transforms.ToPILImage()
        img_A = torch.rand(size=(1,3,256,256)).cpu().clone()
        img_A = img_A.squeeze(0)
        img_A = unloader(img_A)
        img_B = torch.rand(size=(1,3,256,256)).cpu().clone()
        img_B = img_B.squeeze(0)
        img_B = unloader(img_B)
        img_A = self.transform(img_A)
        img_B = self.transform(img_B)
        return {"A": img_A, "B": img_B}

    def __len__(self):
        return self.len

class GANTrainer:
    def __init__(self):
        with open("config.yaml") as f:
            self.cfg = yaml.load(f, Loader=yaml.FullLoader)
        self.generator = GenNN().cuda()
        self.discriminator = DisNN().cuda()
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), 
            lr=self.cfg["lr_rate"], betas=(self.cfg["lr_b1"], self.cfg["lr_b2"]))
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), 
            lr=self.cfg["lr_rate"], betas=(self.cfg["lr_b1"], self.cfg["lr_b1"]))

        self.dataloader = DataLoader(Loaddata(self.cfg),batch_size = self.cfg["batch_size"],
            shuffle = True,num_workers = 4,)


    def train(self):
        self.generator.apply(Weights_Normal)
        self.discriminator.apply(Weights_Normal)
        Tensor = torch.cuda.FloatTensor
        Adv_cGAN = torch.nn.MSELoss().cuda()
        L1_G  = torch.nn.L1Loss().cuda()
        L_vgg = VGG19_PercepLoss().cuda()
        patch = (1, self.cfg["img_height"]//16, self.cfg["img_width"]//16)

        for epoch in range(self.cfg["max_epoch"]):
            for i, batch in enumerate(self.dataloader):
                imgs_distorted = Variable(batch["A"].type(Tensor))
                imgs_groundtruth = Variable(batch["B"].type(Tensor))

                valid = Variable(Tensor(np.ones((imgs_distorted.size(0), *patch))), 
                    requires_grad=False)
                fake = Variable(Tensor(np.zeros((imgs_distorted.size(0), *patch))), 
                    requires_grad=False)

                self.optimizer_D.zero_grad()
                imgs_fake = self.generator(imgs_distorted)
                pred_real = self.discriminator(imgs_groundtruth, imgs_distorted)
                loss_real = Adv_cGAN(pred_real, valid)
                pred_fake = self.discriminator(imgs_fake, imgs_distorted)
                loss_fake = Adv_cGAN(pred_fake, fake)

                loss_D = 0.5*(loss_real+loss_fake)*10.0
                loss_D.backward()
                self.optimizer_D.step()

                self.optimizer_G.zero_grad()
                imgs_fake = self.generator(imgs_distorted)
                pred_fake = self.discriminator(imgs_fake,imgs_distorted)
                loss_GAN =  Adv_cGAN(pred_fake, valid) # GAN loss
                loss_1 = L1_G(imgs_fake, imgs_groundtruth) # similarity loss
                loss_con = L_vgg(imgs_fake, imgs_groundtruth)# content loss
                # Total loss (Section 3.2.1 in the paper)
                loss_G = loss_GAN + 7 * loss_1  + 3 * loss_con 
                loss_G.backward()
                self.optimizer_G.step()

                if not i%10:
                    print("\r[Epoch %d: batch %d] [DLoss: %.3f, GLoss: %.3f, AdvLoss: %.3f]"
                        %(epoch, i,loss_D.item(), loss_G.item(), loss_GAN.item(),))



