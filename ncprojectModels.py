import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class Unet(torch.nn.Module):
    def __init__(self, image_channels, hidden_size=16, n_classes=4):
        super(Unet, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Encoder
        self.conv1_1 = nn.Conv2d(in_channels=image_channels, out_channels=hidden_size, kernel_size=3, stride=1, padding=1)
        self.batch1_1 = nn.BatchNorm2d(hidden_size, affine=True)
        self.conv1_2 = nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=3, stride=1, padding=1)
        self.batch1_2 = nn.BatchNorm2d(hidden_size, affine=True)
        self.pool1 = nn.MaxPool2d(3, 2, padding=1)
        self.dropout1 = nn.Dropout(0.2) 

        self.conv2_1 = nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size*2, kernel_size=3, stride=1, padding=1)
        self.batch2_1 = nn.BatchNorm2d(hidden_size*2, affine=True)
        self.conv2_2 = nn.Conv2d(in_channels=hidden_size*2, out_channels=hidden_size*2, kernel_size=3, stride=1, padding=1)
        self.batch2_2 = nn.BatchNorm2d(hidden_size*2, affine=True)
        self.pool2 = nn.MaxPool2d(3, 2, padding=1)
        self.dropout2 = nn.Dropout(0.2) 

        self.conv3_1 = nn.Conv2d(in_channels=hidden_size*2, out_channels=hidden_size*4, kernel_size=3, stride=1, padding=1)
        self.batch3_1 = nn.BatchNorm2d(hidden_size*4, affine=True)
        self.conv3_2 = nn.Conv2d(in_channels=hidden_size*4, out_channels=hidden_size*4, kernel_size=3, stride=1, padding=1)
        self.batch3_2 = nn.BatchNorm2d(hidden_size*4, affine=True)
        self.pool3 = nn.MaxPool2d(3, 2, padding=1)
        self.dropout3 = nn.Dropout(0.2) 

        # Bottleneck
        self.bottleneck_conv = nn.Conv2d(in_channels=hidden_size*4, out_channels=hidden_size*8, kernel_size=3, stride=1, padding=1)
        self.bottleneck_batch = nn.BatchNorm2d(hidden_size*8, affine=True)

        # Decoder
        self.upsample_3 = nn.ConvTranspose2d(in_channels=hidden_size*8, out_channels=hidden_size*4, kernel_size=2, stride=2)
        self.upconv3_1 = nn.Conv2d(in_channels=hidden_size*8, out_channels=hidden_size*4, kernel_size=3, stride=1, padding=1)
        self.upbatch3_1 = nn.BatchNorm2d(hidden_size*4, affine=True)
        self.upconv3_2 = nn.Conv2d(in_channels=hidden_size*4, out_channels=hidden_size*4, kernel_size=3, stride=1, padding=1)
        self.upbatch3_2 = nn.BatchNorm2d(hidden_size*4, affine=True)
        self.dropout4 = nn.Dropout(0.2) 

        self.upsample_2 = nn.ConvTranspose2d(in_channels=hidden_size*4, out_channels=hidden_size*2, kernel_size=2, stride=2)
        self.upconv2_1 = nn.Conv2d(in_channels=hidden_size*4, out_channels=hidden_size*2, kernel_size=3, stride=1, padding=1)
        self.upbatch2_1 = nn.BatchNorm2d(hidden_size*2, affine=True)
        self.upconv2_2 = nn.Conv2d(in_channels=hidden_size*2, out_channels=hidden_size*2, kernel_size=3, stride=1, padding=1)
        self.upbatch2_2 = nn.BatchNorm2d(hidden_size*2, affine=True)
        self.dropout5 = nn.Dropout(0.2) 

        self.upsample_1 = nn.ConvTranspose2d(in_channels=hidden_size*2, out_channels=hidden_size, kernel_size=2, stride=2)
        self.upconv1_1 = nn.Conv2d(in_channels=hidden_size*2, out_channels=hidden_size, kernel_size=3, stride=1, padding=1)
        self.upbatch1_1 = nn.BatchNorm2d(hidden_size, affine=True)
        self.upconv1_2 = nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=3, stride=1, padding=1)
        self.upbatch1_2 = nn.BatchNorm2d(hidden_size, affine=True)
        self.dropout6 = nn.Dropout(0.2) 

        # Final Layer
        self.conv_out = nn.Conv2d(in_channels=hidden_size, out_channels=n_classes, kernel_size=1, stride=1, padding=0)
        

    def forward(self, x):
        self.enc_layer1 = F.leaky_relu(self.batch1_1(self.conv1_1(x)))
        self.enc_layer1 = F.leaky_relu(self.batch1_2(self.conv1_2(self.enc_layer1)))
        self.enc_layer1_pool = self.dropout1(self.pool1(self.enc_layer1))

        self.enc_layer2 = F.leaky_relu(self.batch2_1(self.conv2_1(self.enc_layer1_pool)))
        self.enc_layer2 = F.leaky_relu(self.batch2_2(self.conv2_2(self.enc_layer2)))
        self.enc_layer2_pool = self.dropout2(self.pool2(self.enc_layer2))

        self.enc_layer3 = F.leaky_relu(self.batch3_1(self.conv3_1(self.enc_layer2_pool)))
        self.enc_layer3 = F.leaky_relu(self.batch3_2(self.conv3_2(self.enc_layer3)))
        self.enc_layer3_pool = self.dropout3(self.pool3(self.enc_layer3))

        self.bottleneck_layer = F.leaky_relu(self.bottleneck_batch(self.bottleneck_conv(self.enc_layer3_pool)))

        self.up3 = torch.cat((self.upsample_3(self.bottleneck_layer), self.enc_layer3), 1)
        self.up3 = F.leaky_relu(self.batch3_1(self.upconv3_1(self.up3)))
        self.up3 = self.dropout4(F.leaky_relu(self.batch3_2(self.upconv3_2(self.up3))))

        self.up2 = torch.cat((self.upsample_2(self.up3), self.enc_layer2), 1)
        self.up2 = F.leaky_relu(self.batch2_1(self.upconv2_1(self.up2)))
        self.up2 = self.dropout5(F.leaky_relu(self.batch2_2(self.upconv2_2(self.up2))))

        self.up1 = torch.cat((self.upsample_1(self.up2), self.enc_layer1), 1)
        self.up1 = F.leaky_relu(self.batch1_1(self.upconv1_1(self.up1)))
        self.up1 = self.dropout6(F.leaky_relu(self.batch1_2(self.upconv1_2(self.up1))))

        self.out = self.conv_out(self.up1)

        return self.out

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
