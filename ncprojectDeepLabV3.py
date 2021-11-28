import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class Block(torch.nn.Module):
    def __init__(self, in_channels, channels, dilation=1):
        super(Block, self).__init__()

        out_channels = channels

        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation, bias=False)
        self.batch1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation, bias=False)
        self.batch2 = nn.BatchNorm2d(channels)

        if in_channels != channels:
            conv = nn.Conv2d(in_channels, channels, kernel_size=1, stride=1, bias=False)
            batch = nn.BatchNorm2d(channels)
            self.downsample = nn.Sequential(conv, batch)
        else:
            self.downsample = nn.Sequential()

    def forward(self, x):
        out = F.relu(self.batch1(self.conv1(x)))
        out = self.batch2(self.conv2(out))
        out = out + self.downsample(x)
        out = F.relu(out)
        return out

class ResNet18(torch.nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        #Use ResNet18
        resnet = models.resnet18()  #18 convolution layers
        # load pretrained model
        resnet.load_state_dict(torch.load("./ResnetModels/resnet18-f37072fd.pth"))
        # remove fully connected layer, avg pool, layer4 and layer5
        self.resnet = nn.Sequential(*list(resnet.children())[:-4])

        blocks = []
        blocks.append(Block(in_channels=128, channels=256, dilation=2))
        blocks.append(Block(in_channels=256, channels=256, dilation=2))
        self.layer4 = nn.Sequential(*blocks)
        blocks = []
        blocks.append(Block(in_channels=256, channels=512, dilation=4))
        blocks.append(Block(in_channels=512, channels=512, dilation=4))
        self.layer5 = nn.Sequential(*blocks)

    def forward(self, x):
        out = self.resnet(x)
        out = self.layer4(out)
        out = self.layer5(out)
        return out

class ASPP(torch.nn.Module):
    def __init__(self, num_classes, in_scale=1):
        super(ASPP, self).__init__()

        self.conv1_1 = nn.Conv2d(in_scale*512, 256, kernel_size=1)
        self.batch1_1 = nn.BatchNorm2d(256)
        self.conv1_2 = nn.Conv2d(in_scale*512, 256, kernel_size=3, stride=1, padding=6, dilation=6)
        self.batch1_2 = nn.BatchNorm2d(256)
        self.conv1_3 = nn.Conv2d(in_scale*512, 256, kernel_size=3, stride=1, padding=12, dilation=12)
        self.batch1_3 = nn.BatchNorm2d(256)
        self.conv1_4 = nn.Conv2d(in_scale*512, 256, kernel_size=3, stride=1, padding=18, dilation=18)
        self.batch1_4 = nn.BatchNorm2d(256)

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.conv2 = nn.Conv2d(in_scale*512, 256, kernel_size=1)
        self.batch2 = nn.BatchNorm2d(256)

        self.conv3 = nn.Conv2d(1280, 256, kernel_size=1)
        self.batch3 = nn.BatchNorm2d(256)

        self.conv4 = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        # get the height and width of feature map for upsampling
        h = x.size()[2]
        w = x.size()[3]

        out1 = F.leaky_relu(self.batch1_1(self.conv1_1(x)))
        out2 = F.leaky_relu(self.batch1_2(self.conv1_2(x)))
        out3 = F.leaky_relu(self.batch1_3(self.conv1_3(x)))
        out4 = F.leaky_relu(self.batch1_4(self.conv1_4(x)))

        out = self.pool(x)
        out = F.leaky_relu(self.batch2(self.conv2(out)))

        # Upsampling step
        out_img = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=True)
        out = torch.cat([out1, out2, out3, out4, out_img], 1)
        out = F.leaky_relu(self.batch3(self.conv3(out)))
        out = self.conv4(out)

        return out


class DeepLabV3(torch.nn.Module):
    def __init__(self, n_classes=4):
        super(DeepLabV3, self).__init__()

        #Load resnet model
        #Pretrained for 18 Convolution layers
        self.resnet = ResNet18()
        # Load ASPP model
        self.aspp = ASPP(num_classes=n_classes,in_scale=1)


    def forward(self, x):
        # get the height and width of feature map for upsampling
        h = x.size()[2]
        w = x.size()[3]

        feature_map = self.resnet(x)
        out = self.aspp(feature_map)
        # Upsampling step
        out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=True)
        return out