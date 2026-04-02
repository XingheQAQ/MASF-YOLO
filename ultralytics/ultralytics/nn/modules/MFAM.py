import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import autopad

class ConvModule(nn.Module):
    """ConvModule的简单实现，包含卷积、归一化、激活函数"""
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size, 
                 stride=1, 
                 padding=0, 
                 dilation=1, 
                 groups=1, 
                 norm_cfg=None, 
                 act_cfg=None):
        super(ConvModule, self).__init__()
        # 定义卷积层
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=False)
        
        # 定义归一化层
        if norm_cfg is not None:
            self.norm = nn.BatchNorm2d(out_channels, momentum=norm_cfg.get('momentum', 0.03), eps=norm_cfg.get('eps', 0.001))
        else:
            self.norm = None

        # 定义激活层
        if act_cfg is not None:
            if act_cfg['type'] == 'ReLU':
                self.activation = nn.ReLU(inplace=True)
            elif act_cfg['type'] == 'SiLU':  # SiLU 是 Swish 的另一种叫法
                self.activation = nn.SiLU(inplace=True)
        else:
            self.activation = None

    def forward(self, x):
        # 前向传播：卷积 -> 归一化 -> 激活
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x
    
from typing import Optional, Sequence  

class MFAM(nn.Module):
    
    def __init__(
            self,
            in_channels: int,
            out_channels: Optional[int] = None,
            kernel_sizes: Sequence[int] = (3, 5, 7, 9),
            dilations: Sequence[int] = (1, 1, 1, 1),
            expansion: float = 1.0,
            add_identity: bool = True,
            norm_cfg: Optional[dict] = dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg: Optional[dict] = dict(type='SiLU'),
    ):
        super().__init__()
        out_channels = out_channels
        hidden_channels = int(in_channels * expansion)

        # 预卷积
        self.pre_conv = ConvModule(in_channels, hidden_channels, 1, 1, 0, 1, norm_cfg=norm_cfg, act_cfg=act_cfg)

        # 3、5小核
        self.dw_conv = ConvModule(hidden_channels, hidden_channels, kernel_sizes[0], 1,
                                  autopad(kernel_sizes[0], None, dilations[0]), dilations[0],
                                  groups=hidden_channels, norm_cfg=None, act_cfg=None)
        self.dw_conv1 = ConvModule(hidden_channels, hidden_channels, kernel_sizes[1], 1,
                                   autopad(kernel_sizes[1], None, dilations[1]), dilations[1],
                                   groups=hidden_channels, norm_cfg=None, act_cfg=None)
        # 将7、9大核分解
        
        self.dw_conv2_w = ConvModule(hidden_channels, hidden_channels, kernel_size=(1,kernel_sizes[2]),
                                   padding=(0,kernel_sizes[2]//2),groups=hidden_channels, norm_cfg=None, act_cfg=None)
        self.dw_conv2_h = ConvModule(hidden_channels, hidden_channels, kernel_size=(kernel_sizes[2],1),
                                   padding=(kernel_sizes[2]//2,0),groups=hidden_channels, norm_cfg=None, act_cfg=None)
        self.dw_conv3_w = ConvModule(hidden_channels, hidden_channels, kernel_size=(1,kernel_sizes[3]),
                                   padding=(0,kernel_sizes[3]//2),groups=hidden_channels, norm_cfg=None, act_cfg=None)
        self.dw_conv3_h = ConvModule(hidden_channels, hidden_channels, kernel_size=(kernel_sizes[3],1),
                                   padding=(kernel_sizes[3]//2,0),groups=hidden_channels, norm_cfg=None, act_cfg=None)
        # 点卷积
        self.pw_conv = ConvModule(hidden_channels, hidden_channels, 1, 1, 0, 1, norm_cfg=norm_cfg, act_cfg=act_cfg)

        # 跳跃连接
        self.add_identity = add_identity and in_channels == out_channels

        # 后卷积
        self.post_conv = ConvModule(hidden_channels, out_channels, 1, 1, 0, 1, norm_cfg=norm_cfg, act_cfg=act_cfg)

    def forward(self, x):
        x = self.pre_conv(x)
        y = x  # 保留输入 x,作为残差连接
        x = x + self.dw_conv(x) + self.dw_conv1(x) + self.dw_conv2_h(self.dw_conv2_w(x)) + self.dw_conv3_h(self.dw_conv3_w(x))
        # 逐点卷积
        x = self.pw_conv(x)

        if self.add_identity:
            x = x + y

        x = self.post_conv(x)
        return x