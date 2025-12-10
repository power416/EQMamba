import torch
import torch.nn as nn
from torch.nn import functional as F
from mamba_ssm import *
from timm.layers import DropPath


class IdentityNTuple(nn.Identity):
    def __init__(self, *args, ntuple: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        assert ntuple >= 1
        self.ntuple = ntuple

    def forward(self, input: torch.Tensor):
        if self.ntuple > 1:
            return (super().forward(input),) * self.ntuple
        else:
            return super().forward(input)
        

class Predict(nn.Module):
    def __init__(self, projection_dim=128, min_size=47, final_dim=16, drop_rate=0.1, num_predictions=3):
        super(Predict, self).__init__()
        conv_layers = []
        mlp_layers = []
        for _ in range(num_predictions):
            conv_layer = nn.Sequential(
                nn.Conv1d(projection_dim, final_dim, 1, 1),
                nn.BatchNorm1d(final_dim),
                nn.GELU()
            )
            conv_layers.append(conv_layer)

            mlp_layer = nn.Sequential(
                nn.Dropout(drop_rate),
                nn.Linear(min_size*final_dim, 1)
            )
            mlp_layers.append(mlp_layer)

        self.post_convs = nn.ModuleList(conv_layers)
        self.mlps = nn.ModuleList(mlp_layers)

    def forward(self, x):
        outputs=[]
        for conv, mlp in zip(self.post_convs, self.mlps):
            x_conv = conv(x)
            x_mlp = mlp(x_conv.flatten(1))
            outputs.append(x_mlp)
            
        return outputs


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, dim, dim2, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv1d(dim, dim, kernel_size, stride=stride, padding=padding, groups=dim)
        self.pointwise = nn.Conv1d(dim, dim2, 1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)

        return x


class Attention(nn.Module):
    def __init__(self, dim, drop_rate):
        super(Attention, self).__init__()
        self.att = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(dim, dim//2, 1),
            nn.GELU(),
            nn.Conv1d(dim//2, dim, 1),
            nn.Sigmoid()
        )
        self.drop = nn.Dropout(drop_rate)

    def forward(self, x):
        x_weight = self.att(x)
        x = self.drop(x * x_weight) + x

        return x


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, drop_rate, has_attn=False):
        super(FeedForward, self).__init__()
        # self.dsconv = DepthwiseSeparableConv(dim, hidden_dim, 5, 1, 2)

        self.dsconv = (
            DepthwiseSeparableConv(dim, hidden_dim, 3, 1, 1)
            if has_attn
            else DepthwiseSeparableConv(dim, hidden_dim, 5, 1, 2)
        )

        self.act = nn.GELU()
        self.drop1 = nn.Dropout(drop_rate)

        self.att = (
            Attention(hidden_dim, drop_rate)
            if has_attn
            else IdentityNTuple()
        )

        self.conv = nn.Conv1d(hidden_dim, dim, 1)
        self.drop2 = nn.Dropout(drop_rate)

    def forward(self, x):
        x = x.permute(0,2,1)
        x1 = self.dsconv(x)
        x = self.drop1(self.act(x1))

        x = self.att(x)

        x = self.drop2(self.conv(x))
        x = x.permute(0,2,1)

        return x
    

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super(PreNorm, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        x = self.norm(x)

        return self.fn(x)+x


class MNet(nn.Module):
    def __init__(self, dim, dim2, depth, drop_rate, has_attn=False):
        super().__init__()
        self.depth = depth
    
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Mamba(d_model=dim, d_state=dim*2, d_conv=4, expand=2)),
                PreNorm(dim, FeedForward(dim, dim2, drop_rate, has_attn))
            ]))

    def forward(self, x):
        x = x.permute(0,2,1)
        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x)

        return x.permute(0,2,1)


class DSConv(nn.Module):
    def __init__(self, dim, hidden_dim, kernel_size, drop_rate):
        super().__init__()
        self.conv_padding_same = (
            (kernel_size - 1) // 2,
            kernel_size - 1 - (kernel_size - 1) // 2,
        )
        self.bn1 = nn.BatchNorm1d(num_features=dim)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(drop_rate)
        self.conv1 = DepthwiseSeparableConv(dim, hidden_dim, kernel_size=kernel_size, padding=0)

        self.bn2 = nn.BatchNorm1d(num_features=hidden_dim)
        self.act2 = nn.GELU()
        self.drop2 = nn.Dropout(drop_rate)
        self.conv2 = DepthwiseSeparableConv(hidden_dim, dim, kernel_size=kernel_size, padding=0)

    def forward(self, x):
        x1 = self.bn1(x)
        x1 = self.act1(x1)
        x1 = self.drop1(x1)
        x1 = F.pad(x1, self.conv_padding_same, "constant", 0)
        x1 = self.conv1(x1)

        x2 = self.bn2(x1)
        x2 = self.act2(x2)
        x2 = self.drop2(x2)
        x2 = F.pad(x2, self.conv_padding_same, "constant", 0)
        x2 = self.conv2(x2)

        return x + x2
    

class MConv(nn.Module):
    def __init__(self, dim, hidden_dim, drop_rate):
        super().__init__()
        self.conv1 = DSConv(dim, hidden_dim, 3, drop_rate)
        self.conv2 = DSConv(dim, hidden_dim, 5, drop_rate)
        self.conv3 = DSConv(dim, hidden_dim, 7, drop_rate)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)

        return x + x3
    

class BaseNet(nn.Module):
    def __init__(self, conv_dim=[3, 16, 32, 64, 128], drop_rate=0.1):
        super(BaseNet, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels=conv_dim[0], out_channels=conv_dim[1], kernel_size=11, stride=8, padding=5),
            nn.Dropout(drop_rate),
            MNet(conv_dim[1], conv_dim[1]//2, 3, drop_rate)
        )
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=conv_dim[1], out_channels=conv_dim[2], kernel_size=7, stride=1, padding=3),
            nn.Dropout(drop_rate),
            nn.MaxPool1d(4),
            MNet(conv_dim[2], conv_dim[2]//2, 3, drop_rate)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=conv_dim[2], out_channels=conv_dim[3], kernel_size=5, stride=1, padding=2),
            nn.Dropout(drop_rate),
            nn.MaxPool1d(2),
            MNet(conv_dim[3], conv_dim[3]//2, 3, drop_rate)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=conv_dim[3], out_channels=conv_dim[4], kernel_size=3, stride=1, padding=1),
            nn.Dropout(drop_rate),
            nn.MaxPool1d(2),
            MNet(conv_dim[4], conv_dim[4]//2, 3, drop_rate, has_attn=True)
        )
        self.conv4 = MConv(conv_dim[4], conv_dim[4]//2, drop_rate)

    def forward(self, x):
        x = self.stem(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        return x


class EQMamba(nn.Module):
    def __init__(
        self, 
        conv_dim=[3, 16, 32, 64, 128], 
        drop_rate=0.1, 
        min_size=46, 
        final_dim=32, 
        out_drop_rate=0.1, 
        num_predictions=3
    ):
        super(EQMamba, self).__init__()

        self.conv = BaseNet(conv_dim=conv_dim, drop_rate=drop_rate)
        self.predict = Predict(projection_dim=conv_dim[-1], min_size=min_size, final_dim=final_dim, drop_rate=out_drop_rate, num_predictions=num_predictions)

    def forward(self, x):
        x = self.conv(x)
        x_o, x_a, x_d = self.predict(x)

        return x_o, x_a, x_d
