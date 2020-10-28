from model import GenNN
import torch

if __name__=="__main__":
    x = torch.rand(size=(8, 3, 256, 256))
    x = x.cuda()
    gnn = GenNN().cuda()
    out = gnn(x)
    print(out.size())