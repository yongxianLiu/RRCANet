
import torch
import torch.nn as nn
from torch.nn import GroupNorm, Module, Sequential, Conv2d, BatchNorm2d, ConvTranspose2d, ReLU, MaxPool2d, Sigmoid, Parameter


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class SBAM(nn.Module):
    def __init__(self, low_planes, high_planes):
        super(SBAM, self).__init__()

        self.pwconv1 = nn.Conv2d(low_planes, low_planes, kernel_size=1)
        self.pwconv2 = nn.Conv2d(high_planes, low_planes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

        self.ca = ChannelAttention(low_planes)


    def forward(self, f_low, f_high):
        high_up = self.pwconv2(f_high)
        high_at = high_up * self.ca(high_up)
        f_low_conv = self.pwconv1(f_low)
        high_at = high_at * self.sigmoid(f_low_conv)

        out = f_low + high_at
        return out


class Res_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1):
        super(Res_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = None

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out



class Block(Module):

    def __init__(self, in_channel, out_channel, conv=None):
        super(Block, self).__init__()
        self.in_channel  = in_channel
        self.out_channel = out_channel
        self.operator    = []
        if conv != None:
            for i in range(len(conv)):
                input_channel, output_channel =conv[i].in_channels, conv[i].out_channels   ##@@@@
                if   input_channel != output_channel:
                    self.shortcut   = nn.Sequential(
                        nn.Conv2d(input_channel, output_channel, kernel_size=1, stride=1),
                        nn.BatchNorm2d(output_channel))

                self.operator.append(conv[i])
                self.operator.append(nn.BatchNorm2d(out_channel))
                self.operator.append(nn.ReLU(inplace=True))
            self.operator = nn.Sequential(*self.operator)

    def forward(self, x):
        if self.operator is None:
            return x

        for i in range(len(self.operator)):
            if i % 6 == 0:
                try:
                    residual = self.shortcut(x)  ##@@@@
                except:
                    residual = x
                out = self.operator[i](x)
                out = self.operator[i + 1](out)
                out = self.operator[i + 2](out)
                out = self.operator[i + 3](out)
                out = self.operator[i + 4](out)

                out += residual
                out = self.operator[i + 5](out)
                x = out
        return x









class RRCANet(nn.Module):
    def __init__(self, num_classes=1, input_channels=1, block=Res_block, num_blocks=[3,2,2,2], iterations=2, multiplier=1.0, num_layers=4, integrate='True', deep_supervision='False'):
        super(RRCANet, self).__init__()
        self.iterations = iterations
        self.multiplier = multiplier
        self.num_layers = num_layers
        self.integrate = integrate

        self.deep_supervision = deep_supervision

        #生成通道数
        self.filters_list = [int(8 * (2 ** i) * self.multiplier) for i in range(self.num_layers)]

        #预处理块，不参与循环，8*128*128

        self.conv0_0 = self._make_layer2(block, input_channels, self.filters_list[0])

        self.pool = nn.MaxPool2d(2, 2)


        reuse_convs = []  # encoder复用的卷积核
        self.encoders = []  # 由encoder构成的列表。由于encoder的一部分不参与循环，因此每个encoder是一个元组(两个CONV的Sequential, DOWN)
        reuse_deconvs = []  # decoder复用的卷积
        self.decoders = []
        self.SBAM =[]

        for iteration in range(self.iterations+1):
            for layer in range(self.num_layers):
                if layer == 0:
                    in_channel = self.filters_list[0]
                else:
                    in_channel = self.filters_list[layer-1]

                out_channel = self.filters_list[layer]
                #  创建encoders模型
                if iteration == 0:
                    conv_block = self._make_layer(in_channel, out_channel, num_blocks[layer])
                    #conv1 = nn.Conv2d(out_channel, out_channel, kernel_size=1, groups=mid_channel)
                    reuse_convs.append(conv_block)
                conv_block = reuse_convs[layer]
                #  创建encoder
                #  首先构造残差块
                convs = Block(in_channel, out_channel, conv_block)

                self.add_module("iteration{0}_layer{1}_encoder_convs".format(iteration, layer), convs)
                self.encoders.append(convs)

            for layer in range(self.num_layers-1):
                low_channels = self.filters_list[self.num_layers - 2 - layer]
                high_channels = self.filters_list[self.num_layers - 1 - layer]

                decoder_SBAM = SBAM(low_channels, high_channels)
                self.add_module("iteration{0}_layer{1}_decoders_SBAM".format(iteration, layer), decoder_SBAM)
                self.SBAM.append(decoder_SBAM)


        #  创建middle层
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        if (self.integrate == 'True' and self.iterations!=0 and self.deep_supervision == 'False'):
            self.post_transform_conv_block = self._make_layer2(block, self.filters_list[0]*(self.iterations+1), self.filters_list[0])
        else:
            self.post_transform_conv_block = self._make_layer2(block, self.filters_list[0], self.filters_list[0])

        self.final = nn.Conv2d(self.filters_list[0], num_classes, kernel_size=1)

    def _make_layer(self, input_channels, output_channels, num_blocks=1):
        try:
            layers    = []
            layers.append(Conv2d(input_channels, output_channels, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1), bias=True))
            for i in range(2*num_blocks-1):
                layers.append(Conv2d(output_channels,  output_channels, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1), bias=True))
            return nn.Sequential(*layers)
        except:
            return None

    def _make_layer2(self, block, input_channels, output_channels, num_blocks=1):
        layers = []
        layers.append(block(input_channels, output_channels))
        for i in range(num_blocks - 1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)

    def forward(self, input):
        enc = [None for i in range(self.num_layers-1)]
        dec = [None for i in range(self.num_layers-1)]
        all_output = [None for i in range(self.iterations+1)]
        x0_0 = self.conv0_0(input)
        e_i = 0
        d_i = 0
        for iteration in range(self.iterations+1):
            for layer in range(self.num_layers):
                if (iteration==0 and layer == 0):
                    x_in = x0_0
                x_in = self.encoders[e_i](x_in)
                if layer != (self.num_layers-1):
                    enc[layer] = x_in
                    x_in = self.pool(x_in)
                e_i = e_i + 1

            for layer in range(self.num_layers-1):
                    x_in = self.up(x_in)
                    x_in = self.SBAM[d_i](enc[-1-layer], x_in)
                    dec[layer] = x_in
                    d_i = d_i + 1
            all_output[iteration] = x_in
        if self.deep_supervision == 'True':
            out = [None for i in range(self.iterations + 1)]
            for i in range(len(all_output)):
                x_in = self.post_transform_conv_block(all_output[i])
                output = self.final(x_in).sigmoid()
                out[i] = output
        else:
            if self.iterations and self.integrate == 'True':
                x_in = torch.cat(all_output, dim=1)
            x_in = self.post_transform_conv_block(x_in)
            out = self.final(x_in).sigmoid()
        return out