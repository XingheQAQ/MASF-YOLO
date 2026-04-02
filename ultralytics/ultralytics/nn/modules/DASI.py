import torch
import torch.nn as nn

class Bag(nn.Module):
    def __init__(self):
        super(Bag, self).__init__()
    def forward(self, p, i, d):
        edge_att = torch.sigmoid(d)
        return edge_att * p + (1 - edge_att) * i

class DASI(nn.Module):
    def __init__(self, in_features) -> None:
        super().__init__()

        self.bag = Bag()
        self.tail_conv = nn.Conv2d(in_features,in_features,1 )
        self.bns = nn.BatchNorm2d(in_features)
        self.gelu = nn.GELU()

    def forward(self, x):
        x_low, x_mid ,x_high = x
        x_identity = x_mid

        x_low = torch.chunk(x_low, 4, dim=1)
        x_mid = torch.chunk(x_mid, 4, dim=1)
        x_high = torch.chunk(x_high, 4, dim=1)
        
        x0 = self.bag(x_low[0], x_high[0], x_mid[0])
        x1 = self.bag(x_low[1], x_high[1], x_mid[1])
        x2 = self.bag(x_low[2], x_high[2], x_mid[2])
        x3 = self.bag(x_low[3], x_high[3], x_mid[3])
        x = torch.cat((x0, x1, x2, x3), dim=1)
        
        x = x + x_identity
        x = self.tail_conv(x)
        x = self.bns(x)
        x = self.gelu(x)
        
        return x 