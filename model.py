
import torch
import torch.nn as nn
from resnet import resnet50
import numpy as np
import cv2

def save_feats_mean(x, size=(256, 256)):
    b, c, h, w = x.shape
    with torch.no_grad():
        x = x.detach().cpu().numpy()
        x = np.transpose(x[0], (1, 2, 0))
        x = np.mean(x, axis=-1)
        x = x/np.max(x)
        x = x * 255.0
        x = x.astype(np.uint8)

        if h != size[1]:
            x = cv2.resize(x, size)

        x = cv2.applyColorMap(x, cv2.COLORMAP_JET)
        x = np.array(x, dtype=np.uint8)
        return x

def get_mean_attention_map(x):
    x = torch.mean(x, axis=1)
    x = torch.unsqueeze(x, 1)
    x = x / torch.max(x)
    return x

class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.relu = nn.ReLU()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c)
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_c)
        )

    def forward(self, inputs):
        x1 = self.conv(inputs)
        x2 = self.shortcut(inputs)
        x = self.relu(x1 + x2)
        return x

class DilatedConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.c1 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )

        self.c2 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=3, dilation=3),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )

        self.c3 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=6, dilation=6),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )

        self.c4 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=9, dilation=9),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )

        self.c5 = nn.Sequential(
            nn.Conv2d(out_c*4, out_c, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )

    def forward(self, inputs):
        x1 = self.c1(inputs)
        x2 = self.c2(inputs)
        x3 = self.c3(inputs)
        x4 = self.c4(inputs)
        x = torch.cat([x1, x2, x3, x4], axis=1)
        x = self.c5(x)
        return x

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x0 = x
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return x0 * self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x0 = x
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return x0 * self.sigmoid(x)

class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.r1 = ResidualBlock(in_c[0]+in_c[1], out_c)
        self.r2 = ResidualBlock(out_c, out_c)

        self.ca = ChannelAttention(out_c)
        self.sa = SpatialAttention()

    def forward(self, x, s):
        x = self.up(x)
        x = torch.cat([x, s], axis=1)
        x = self.r1(x)
        x = self.r2(x)

        x = self.ca(x)
        x = self.sa(x)
        return x

class RUPNet(nn.Module):
    def __init__(self):
        super().__init__()

        """ ResNet50 """
        backbone = resnet50()
        self.layer0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.layer1 = nn.Sequential(backbone.maxpool, backbone.layer1)
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3

        """ Dilated Conv + Pooling """
        self.r1 = nn.Sequential(DilatedConv(64, 64), nn.MaxPool2d((8, 8)))
        self.r2 = nn.Sequential(DilatedConv(256, 64), nn.MaxPool2d((4, 4)))
        self.r3 = nn.Sequential(DilatedConv(512, 64), nn.MaxPool2d((2, 2)))
        self.r4 = DilatedConv(1024, 64)

        """ Decoder """
        self.d1 = DecoderBlock([256, 512], 256)
        self.d2 = DecoderBlock([256, 256], 128)
        self.d3 = DecoderBlock([128, 64], 64)
        self.d4 = DecoderBlock([64, 3], 32)

        """  """

        """ Output """
        self.y = nn.Conv2d(32, 1, kernel_size=1, padding=0)

    def forward(self, x, heatmap=None):
        """ ResNet50 """
        s0 = x
        s1 = self.layer0(s0)    ## [-1, 64, h/2, w/2]
        s2 = self.layer1(s1)    ## [-1, 256, h/4, w/4]
        s3 = self.layer2(s2)    ## [-1, 512, h/8, w/8]
        s4 = self.layer3(s3)    ## [-1, 1024, h/16, w/16]

        """ Dilated Conv + Pooling """
        r1 = self.r1(s1)
        r2 = self.r2(s2)
        r3 = self.r3(s3)
        r4 = self.r4(s4)

        rx = torch.cat([r1, r2, r3, r4], axis=1)

        """ Decoder """
        d1 = self.d1(rx, s3)
        d2 = self.d2(d1, s2)
        d3 = self.d3(d2, s1)
        d4 = self.d4(d3, s0)

        y = self.y(d4)

        if heatmap != None:
            hmap = save_feats_mean(d4)
            return hmap, y
        else:
            return y

if __name__ == "__main__":
    x = torch.randn((8, 3, 256, 256))
    model = RUPNet()
    y = model(x)
    print(y.shape)
    
    from ptflops import get_model_complexity_info
    flops, params = get_model_complexity_info(model, input_res=(3, 256, 256), as_strings=True, print_per_layer_stat=False)
    print('      - Flops:  ' + flops)
    print('      - Params: ' + params)
