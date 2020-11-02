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
import random
import time
import datetime
import sys
import numpy as np
import itertools

class ReplayBuffer():
    def __init__(self, max_size=50):
        assert (max_size > 0), 'Empty buffer or trying to create a black hole. Be careful.'
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0,1) > 0.5:
                    i = random.randint(0, self.max_size-1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))

class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant(m.bias.data, 0.0)

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        conv_block = [nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features,3),
                        nn.InstanceNorm2d(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features)]
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super(Generator, self).__init__()
        model = [nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)]
        
        in_features = 64
        out_features = in_features*2
        for _ in range(2):
            model += [nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features*2
        
        for _ in range(n_residual_blocks):
            model +=[ResidualBlock(in_features)]
        
        out_features = in_features//2
        for _ in range(2):
            model += [nn.ConvTranspose2d(in_features, out_features, 3, stride=2,
                        padding=1, output_padding=1),nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features//2

        model+= [nn.ReflectionPad2d(3),nn.Conv2d(64, output_nc, 7),nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()

        model = [nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True)]
        model += [nn.Conv2d(64, 128, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(128),
                    nn.LeakyReLU(0.2, inplace=True)]
        model += [nn.Conv2d(128, 256, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(128),
                    nn.LeakyReLU(0.2, inplace=True)]
        model += [nn.Conv2d(256,512,4, padding=1),
                    nn.InstanceNorm2d(512),
                    nn.LeakyReLU(0.2,inplace=True)]
        model += [nn.Conv2d(512,1, 4, padding=1)]
        self.model = nn.Sequential(*model)


    def forward(self, x):
        x = self.model(x)
        return F.avg_pool2d(x,x.size()[2:]).view(x.size()[0])


class Loaddata(Dataset):
    def __init__(self,cfg):
        transforms_ = [
            transforms.Resize(int(cfg["size"]*1.12), Image.BICUBIC),
            transforms.RandomCrop(cfg["size"]),
            transforms.RandomHorizontalFlip(),
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


class CycleGANTrainer:
    def __init__(self):
        with open("config.yaml") as f:
            self.cfg = yaml.load(f, Loader=yaml.FullLoader)
        self.netG_A2B = Generator(self.cfg["input_nc"], self.cfg["output_nc"]).cuda()
        self.netG_B2A = Generator(self.cfg["output_nc"], self.cfg["input_nc"]).cuda()
        self.netD_A = Discriminator(self.cfg["input_nc"]).cuda()
        self.netD_B = Discriminator(self.cfg["output_nc"]).cuda()

        self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A2B.parameters(), 
                        self.netG_B2A.parameters()),lr=self.cfg["lr"], betas=(0.5, 0.999))
        self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(), lr=self.cfg["lr"],
                        betas=(0.5, 0.999))
        self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(), lr=self.cfg["lr"], 
                        betas=(0.5, 0.999))
        self.lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(self.optimizer_G, lr_lambda
                        = LambdaLR(self.cfg["n_epochs"], self.cfg["epoch"],self.cfg["decay_epoch"]).step)
        self.lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(self.optimizer_D_A, lr_lambda
                        = LambdaLR(self.cfg["n_epochs"], self.cfg["epoch"],self.cfg["decay_epoch"]).step)
        self.lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(self.optimizer_D_B, lr_lambda
                        = LambdaLR(self.cfg["n_epochs"], self.cfg["epoch"],self.cfg["decay_epoch"]).step)


        self.dataloader = DataLoader(Loaddata(self.cfg),batch_size = self.cfg["batchSize"],
            shuffle = True,num_workers = 4,)

    def train(self):
        criterion_GAN = torch.nn.MSELoss()
        criterion_cycle = torch.nn.L1Loss()
        criterion_identity = torch.nn.L1Loss()
        Tensor = torch.cuda.FloatTensor

        target_real = Variable(Tensor(self.cfg["batchSize"]).fill_(1.0), requires_grad=False)
        target_fake = Variable(Tensor(self.cfg["batchSize"]).fill_(0.0), requires_grad=False)

        fake_A_buffer = ReplayBuffer()
        fake_B_buffer = ReplayBuffer()

        for epoch in range(self.cfg["n_epochs"]):
            for i, batch in enumerate(self.dataloader):
                real_A = Variable(batch["A"].type(Tensor))
                real_B = Variable(batch["B"].type(Tensor))

                self.optimizer_G.zero_grad()
                same_B = self.netG_A2B(real_B)
                loss_identity_B = criterion_identity(same_B, real_B)*5.0
                same_A = self.netG_B2A(real_A)
                loss_identity_A = criterion_identity(same_A, real_A)*5.0

                fake_B = self.netG_A2B(real_A)
                pred_fake = self.netD_B(fake_B)
                loss_GAN_A2B = criterion_GAN(pred_fake, target_real)

                fake_A = self.netG_B2A(real_B)
                pred_fake = self.netD_A(fake_A)
                loss_GAN_B2A = criterion_GAN(pred_fake, target_real)

                recovered_A = self.netG_B2A(fake_B)
                loss_cycle_ABA = criterion_cycle(recovered_A, real_A)*10.0

                recovered_B = self.netG_A2B(fake_A)
                loss_cycle_BAB = criterion_cycle(recovered_B, real_B)*10.0

                loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B+loss_GAN_B2A \
                            + loss_cycle_ABA + loss_cycle_BAB
                loss_G.backward()

                self.optimizer_G.step()

                self.optimizer_D_A.zero_grad()
                pred_real = self.netD_A(real_A)
                loss_D_real = criterion_GAN(pred_real, target_real)

                fake_A = fake_A_buffer.push_and_pop(fake_A)
                pred_fake = self.netD_A(fake_A.detach())
                loss_D_fake = criterion_GAN(pred_fake, target_fake)

                loss_D_A = (loss_D_real + loss_D_fake) * 0.5 
                loss_D_A.backward()
                self.optimizer_D_A.step()

                self.optimizer_D_B.zero_grad()
                pred_real = self.netD_B(real_B)
                loss_D_real = criterion_GAN(pred_real, target_real)
                fake_B = fake_B_buffer.push_and_pop(fake_B)
                pred_fake = self.netD_B(fake_B.detach())
                loss_D_fake = criterion_GAN(pred_fake, target_fake)
                loss_D_B = (loss_D_real+loss_D_fake)*0.5 
                loss_D_B.backward()
                self.optimizer_D_B.step()

                if(i%5==0):
                    print(
                    {'loss_G': loss_G.item(), 
                    'loss_G_identity': loss_identity_A.item() + loss_identity_B.item(), 
                    'loss_G_GAN': loss_GAN_A2B.item() + loss_GAN_B2A.item(),
                    'loss_G_cycle': loss_cycle_ABA.item() + loss_cycle_BAB.item(), 
                    'loss_D': loss_D_A.item() + loss_D_B.item()}
                    )

            self.lr_scheduler_G.step()
            self.lr_scheduler_D_A.step()
            self.lr_scheduler_D_B.step()


                