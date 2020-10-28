from model import GenNN, DisNN,GANTrainer
import torch


def test_GenNN():
    x = torch.rand(size=(8, 3, 256, 256))
    x = x.cuda()
    gnn = GenNN().cuda()
    out = gnn(x)
    print(out.size())

def print_GenNN_param():
    gnn = GenNN()
    for name, param in gnn.named_parameters():
        print(name, '      ', param.size())

def test_DisNN():
    ## 输出大小为1,1,16,16
    x = torch.rand(size=(1,3,256,256))
    y = torch.rand(size=(1,3,256,256))
    x = x.cuda()
    y = y.cuda()
    dnn = DisNN().cuda()
    out = dnn(x,y)
    print(out.size())

def print_DisNN_param():
    dnn = DisNN()
    for name, param in dnn.named_parameters():
        print(name, '      ', param.size())


if __name__=="__main__":
    ## print_DisNN_param()
    gan_trainer = GANTrainer()
    gan_trainer.train()
