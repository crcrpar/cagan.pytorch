import torch
import torch.nn as nn


def get_deconv(deconv_type, in_ch, out_ch, bias=False):
    if deconv_type == 'deconv':
        return nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, bias=bias)
    else:
        layer = nn.Sequential((
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, bias=bias)))
        return layer


class Generator(nn.Module):

    batch_size = 16

    def __init__(self, deconv='upconv', relu='relu6', bias=True, mask_norm=1):
        super(Generator, self).__init__()
        self.mask_norm = mask_norm
        _conv = {'kernel_size': 3, 'stride': 2, 'bias': bias}
        # embedding
        self.pad1 = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(9, 64, kernel_size=3, stride=2, bias=False)
        self.pad2 = nn.ReflectionPad2d(1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, bias=False)
        self.norm2 = nn.InstanceNorm2d(128)
        self.pad3 = nn.ReflectionPad2d(1)
        self.conv3 = nn.Conv2d(128, 256, **_conv)
        self.norm3 = nn.InstanceNorm2d(256)
        self.pad4 = nn.ReflectionPad2d(1)
        self.conv4 = nn.Conv2d(256, 512, **_conv)
        self.norm4 = nn.InstanceNorm2d(512)
        # generating
        self.pad5 = nn.ReflectionPad2d(1)
        self.deconv5 = get_deconv(deconv, 512, 512, bias)
        self.norm5 = nn.InstanceNorm2d(512)
        self.padd6 = nn.ReflectionPad2d(1)
        self.deconv6 = get_deconv(deconv, 512 + 256, 256, bias)
        self.norm6 = nn.InstanceNorm2d(256)
        self.pad7 = nn.ReflectionPad2d(1)
        self.deconv7 = get_deconv(deconv, 256 + 128, 128, bias)
        self.pad8 = nn.ReflectionPad2d(1)
        self.deconv8 = get_deconv(deconv, 128 + 64, 4, bias)
        # Activations
        if relu == 'relu6':
            self.relu = nn.ReLU6(inplace=True)
        else:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # use skip connections. Here, simply conncatenate channels.
        # input shape: 128x96
        # l1 output: 64x48
        # l2 output: 32x24
        # l3 output: 16x12
        # l4 output: 8x6
        # l5 output: 16x12 (l3)
        # l6 output: 32x24 (l2)
        # l7 output: 64x48 (l1)
        # l8 output: 128x96
        stored_downsampled = dict()
        h = self.relu(self.conv1(self.pad1(x)))
        h1 = h
        stored_downsampled['l1'] = h[:, -6:, Ellipsis]
        h = self.relu(self.norm2(self.conv2(self.pad2(h))))
        h2 = h
        stored_downsampled['l2'] = h[:, -6:, Ellipsis]
        h = self.relu(self.norm3(self.conv3(self.pad3(h))))
        h3 = h
        stored_downsampled['l3'] = h[:, -6:, Ellipsis]
        h = self.relu(self.norm4(self.conv4(self.pad4(h))))
        stored_downsampled['l4'] = h[:, -6:, Ellipsis]
        h = self.relu(self.norm5(self.deconv5(self.pad5(h))))
        stored_downsampled['l5'] = h[:, -6:, Ellipsis]
        h = torch.cat([h, h3], dim=1)
        h = self.relu(self.norm6(self.deconv6(self.pad6(h))))
        stored_downsampled['l6'] = h[:, -6:, Ellipsis]
        h = torch.cat([h, h2], dim=1)
        h = self.relu(self.norm7(self.deconv7(self.pad7(h))))
        stored_downsampled['l7'] = h[:, -6:, Ellipsis]
        h = torch.cat([h, h1], dim=1)
        y = self.relu(self.deconv8(self.pad8(h)))
        return y, stored_downsampled

    def render(self, x, y):
        mask = y[:, 0, Ellipsis]
        y = y[:, 1:, Ellipsis]
        return mask * y + (1 - mask) * x, torch.norm(mask, p=self.mask_norm)


class Discriminator(nn.Module):

    batch_size = 16

    def __init__(self, relu='relu6', bias=True):
        super(Discriminator, self).__init__()
        _conv = {'kernel_size': 3, 'stride': 2, 'bias': bias}
        self.pad1 = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(6, 64, **_conv)
        self.pad2 = nn.ReflectionPad2d(1)
        self.conv2 = nn.Conv2d(64, 128, **_conv)
        self.norm2 = nn.InstanceNorm2d(128)
        self.pad3 = nn.ReflectionPad2d(1)
        self.conv3 = nn.Conv2d(128, 256, **_conv)
        self.norm3 = nn.InstanceNorm2d(256)
        self.pad4 = nn.ReflectionPad2d(1)
        self.conv4 = nn.Conv2d(256, 512, **_conv)
        self.prob = nn.Linear(512 * 8 * 6, 1, bias=bias)
        # Activations
        if relu == 'relu6':
            self.relu = nn.ReLU6(inplace=True)
        else:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        h = self.relu(self.conv1(self.pad1(x)))
        h = self.relu(self.norm2(self.conv2(self.pad2(h))))
        h = self.relu(self.norm3(self.conv3(self.pad3(h))))
        y = self.conv4(h)
        return y
