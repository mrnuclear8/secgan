import torch
from torch import nn
from half_inst import half_instance

class convblock(nn.Module):
    def __init__(self, inc, ouc, K=3, S=1, norm=half_instance):
        super(convblock, self).__init__()
        self.pad = nn.ReflectionPad2d(K // 2),
        self.conv = nn.Conv2d(inc, ouc, kernel_size=K, stride=S)
        if norm != None:
            self.norm = norm(ouc)
        else:
            self.norm = None
        self.act = nn.LeakyReLU(0.2, True)
        self.fwd = nn.Sequential(
            self.pad,
            self.conv,
            self.norm,
            self.act,
        )
    def forawd(self, x):
        x = self.fwd(x)

        return x
class ResBlock(nn.Module):
    def __init__(self, inc, ouc, K=3, S=1, norm=half_instance):
        self.stride = S
        self.conv1 = convblock(inc, ouc, K, S, norm)
        self.conv2 = convblock(ouc, ouc, K, S, norm = None)

    def forward(self, x):
        res = x
        x = self.conv1(x)
        x = self.conv2(x)
        if self.stride == 1:
            x = x + res
        return x
class simple_unet(nn.Module):
    def __init__(self):
        super(simple_unet, self).__init__()
        self.stem =


if __name__ == "__main__":
    x = torch.rand([1, 3, 224, 224])
    print(x[:, :, ::2, ::2].shape)
    nn.PixelShuffle