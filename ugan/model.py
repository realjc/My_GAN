import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import models
from torch.utils.data import Dataset
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.autograd as autograd
import torchvision.transforms as transforms
from PIL import Image
import yaml
import numpy as np 
import torch.nn.functional as F

class UNetDown(nn.Module):
    """ Standard UNet down-sampling block 
    """
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    """ Standard UNet up-sampling block
    """
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)
        return x

class GeneratorUNet(nn.Module):
    """ Standard UNet generator with skip connections used by  
        - Pix2Pix (https://phillipi.github.io/pix2pix/)
        - UGAN (https://github.com/cameronfabbri/Underwater-Color-Correction)  
    """
    def __init__(self, in_channels=3, out_channels=3):
        super(GeneratorUNet, self).__init__()

        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5)
        self.down7 = UNetDown(512, 512, dropout=0.5)
        self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5)

        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 512, dropout=0.5)
        self.up4 = UNetUp(1024, 512, dropout=0.5)
        self.up5 = UNetUp(1024, 256)
        self.up6 = UNetUp(512, 128)
        self.up7 = UNetUp(256, 64)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, out_channels, 4, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)

        return self.final(u7)


class Discriminator(nn.Module):
    def __init__(self, in_channels = 3):
        super(Discriminator,self).__init__()
        def discriminator_block(in_filters, out_filters, normalization=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        self.model = nn.Sequential(
            *discriminator_block(in_channels, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128,256),
            *discriminator_block(256,512),
            nn.ZeroPad2d((1,0,1,0)),
            nn.Conv2d(512,1,4,padding=1,bias=False)
        )

    def forward(self, img):
        return self.model(img)



class Gradient_Penalty(nn.Module):
    """ Calculates the gradient penalty loss for WGAN GP
    """
    def __init__(self, cuda=True):
        super(Gradient_Penalty, self).__init__()
        self.Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    def forward(self, D, real, fake):
        # Random weight term for interpolation between real and fake samples
        eps = self.Tensor(np.random.random((real.size(0), 1, 1, 1)))
        # Get random interpolation between real and fake samples
        interpolates = (eps * real + ((1 - eps) * fake)).requires_grad_(True)
        d_interpolates = D(interpolates)
        fake = autograd.Variable(self.Tensor(d_interpolates.shape).fill_(1.0), requires_grad=False)
        # Get gradient w.r.t. interpolates
        gradients = autograd.grad(outputs=d_interpolates,
                                  inputs=interpolates,
                                  grad_outputs=fake,
                                  create_graph=True,
                                  retain_graph=True,
                                  only_inputs=True,)[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

class Gradient_Difference_Loss(nn.Module):
    def __init__(self, alpha=1, chans=3, cuda=True):
        super(Gradient_Difference_Loss, self).__init__()
        self.alpha = alpha
        self.chans = chans
        Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        SobelX = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
        SobelY = [[1, 2, -1], [0, 0, 0], [1, 2, -1]]
        self.Kx = Tensor(SobelX).expand(self.chans, 1, 3, 3)
        self.Ky = Tensor(SobelY).expand(self.chans, 1, 3, 3)

    def get_gradients(self, im):
        gx = F.conv2d(im, self.Kx, stride=1, padding=1, groups=self.chans)
        gy = F.conv2d(im, self.Ky, stride=1, padding=1, groups=self.chans)
        return gx, gy

    def forward(self, pred, true):
        # get graduent of pred and true
        gradX_true, gradY_true = self.get_gradients(true)
        grad_true = torch.abs(gradX_true) + torch.abs(gradY_true)
        gradX_pred, gradY_pred = self.get_gradients(pred)
        grad_pred_a = torch.abs(gradX_pred)**self.alpha + torch.abs(gradY_pred)**self.alpha
        # compute and return GDL
        return 0.5 * torch.mean(grad_true - grad_pred_a)

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


class UGANTrainer:
    def __init__(self):
        with open("config.yaml") as f:
            self.cfg = yaml.load(f, Loader=yaml.FullLoader)
        self.generator = GeneratorUNet().cuda()
        self.discriminator = Discriminator().cuda()
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=self.cfg["lr_rate"])
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.cfg["lr_rate"])
        self.dataloader = DataLoader(Loaddata(self.cfg),batch_size = self.cfg["batch_size"],
            shuffle = True,num_workers = 4,)

    def train(self):
        L1_G  = torch.nn.L1Loss().cuda()
        L1_gp = Gradient_Penalty().cuda()
        L_gdl = Gradient_Difference_Loss().cuda()
        Tensor = torch.cuda.FloatTensor

        for epoch in range(self.cfg["num_epochs"]):
            for i, batch in enumerate(self.dataloader):
                # Model inputs
                imgs_distorted = Variable(batch["A"].type(Tensor))
                imgs_good_gt = Variable(batch["B"].type(Tensor))

                ## Train Discriminator
                self.optimizer_D.zero_grad()
                imgs_fake = self.generator(imgs_distorted)
                pred_real = self.discriminator(imgs_good_gt)
                pred_fake = self.discriminator(imgs_fake)
                loss_D = -torch.mean(pred_real) + torch.mean(pred_fake) # wgan 
                gradient_penalty = L1_gp(self.discriminator, imgs_good_gt.data, imgs_fake.data)
                loss_D += self.cfg["lambda_gp"] * gradient_penalty # Eq.2 paper 
                loss_D.backward()
                self.optimizer_D.step()

                self.optimizer_G.zero_grad()
                ## Train Generator at 1:num_critic rate 
                if i % self.cfg["num_critic"] == 0:
                    imgs_fake = self.generator(imgs_distorted)
                    pred_fake = self.discriminator(imgs_fake.detach())
                    loss_gen = -torch.mean(pred_fake)
                    loss_1 = L1_G(imgs_fake, imgs_good_gt)
                    loss_gdl = L_gdl(imgs_fake, imgs_good_gt)
                    # Total loss: Eq.6 in paper
                    loss_G = loss_gen + self.cfg["lambda_1"] * loss_1 + self.cfg["lambda_2"] * loss_gdl   
                    loss_G.backward()
                    self.optimizer_G.step()

                ## Print log
                if not i%50:
                    print("\r[Epoch %d/%d: batch %d/%d] [DLoss: %.3f, GLoss: %.3f]"
                                    %(
                                        epoch, self.cfg["num_epochs"], i, len(self.dataloader),
                                        loss_D.item(), loss_G.item(),
                                    )
                    )