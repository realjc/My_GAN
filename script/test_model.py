from model import GenNN

if __name__=="__main__":
    x = torch.rand(size=(8, 3, 256, 256))
    gnn = GenNN().cuda()
    out = gnn(x)
    print(out.size())