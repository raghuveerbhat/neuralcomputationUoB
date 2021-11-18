import torch
import torch.nn as nn
import torch.nn.functional as F


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

        self.conv2_1 = nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size*2, kernel_size=3, stride=1, padding=1)
        self.batch2_1 = nn.BatchNorm2d(hidden_size*2, affine=True)
        self.conv2_2 = nn.Conv2d(in_channels=hidden_size*2, out_channels=hidden_size*2, kernel_size=3, stride=1, padding=1)
        self.batch2_2 = nn.BatchNorm2d(hidden_size*2, affine=True)
        self.pool2 = nn.MaxPool2d(3, 2, padding=1)

        self.conv3_1 = nn.Conv2d(in_channels=hidden_size*2, out_channels=hidden_size*4, kernel_size=3, stride=1, padding=1)
        self.batch3_1 = nn.BatchNorm2d(hidden_size*4, affine=True)
        self.conv3_2 = nn.Conv2d(in_channels=hidden_size*4, out_channels=hidden_size*4, kernel_size=3, stride=1, padding=1)
        self.batch3_2 = nn.BatchNorm2d(hidden_size*4, affine=True)
        self.pool3 = nn.MaxPool2d(3, 2, padding=1)

        # Bottleneck
        self.bottleneck_conv = nn.Conv2d(in_channels=hidden_size*4, out_channels=hidden_size*8, kernel_size=3, stride=1, padding=1)
        self.bottleneck_batch = nn.BatchNorm2d(hidden_size*8, affine=True)

        # Decoder
        self.upsample_3 = nn.ConvTranspose2d(in_channels=hidden_size*8, out_channels=hidden_size*4, kernel_size=2, stride=2)
        self.upconv3_1 = nn.Conv2d(in_channels=hidden_size*8, out_channels=hidden_size*4, kernel_size=3, stride=1, padding=1)
        self.upbatch3_1 = nn.BatchNorm2d(hidden_size*4, affine=True)
        self.upconv3_2 = nn.Conv2d(in_channels=hidden_size*4, out_channels=hidden_size*4, kernel_size=3, stride=1, padding=1)
        self.upbatch3_2 = nn.BatchNorm2d(hidden_size*4, affine=True)

        self.upsample_2 = nn.ConvTranspose2d(in_channels=hidden_size*4, out_channels=hidden_size*2, kernel_size=2, stride=2)
        self.upconv2_1 = nn.Conv2d(in_channels=hidden_size*4, out_channels=hidden_size*2, kernel_size=3, stride=1, padding=1)
        self.upbatch2_1 = nn.BatchNorm2d(hidden_size*2, affine=True)
        self.upconv2_2 = nn.Conv2d(in_channels=hidden_size*2, out_channels=hidden_size*2, kernel_size=3, stride=1, padding=1)
        self.upbatch2_2 = nn.BatchNorm2d(hidden_size*2, affine=True)

        self.upsample_1 = nn.ConvTranspose2d(in_channels=hidden_size*2, out_channels=hidden_size, kernel_size=2, stride=2)
        self.upconv1_1 = nn.Conv2d(in_channels=hidden_size*2, out_channels=hidden_size, kernel_size=3, stride=1, padding=1)
        self.upbatch1_1 = nn.BatchNorm2d(hidden_size, affine=True)
        self.upconv1_2 = nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=3, stride=1, padding=1)
        self.upbatch1_2 = nn.BatchNorm2d(hidden_size, affine=True)

        # Final Layer
        self.conv_out = nn.Conv2d(in_channels=hidden_size, out_channels=n_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        self.enc_layer1 = F.leaky_relu(self.batch1_1(self.conv1_1(x)))
        self.enc_layer1 = F.leaky_relu(self.batch1_2(self.conv1_2(self.enc_layer1)))
        self.enc_layer1_pool = self.pool1(self.enc_layer1)

        self.enc_layer2 = F.leaky_relu(self.batch2_1(self.conv2_1(self.enc_layer1_pool)))
        self.enc_layer2 = F.leaky_relu(self.batch2_2(self.conv2_2(self.enc_layer2)))
        self.enc_layer2_pool = self.pool2(self.enc_layer2)

        self.enc_layer3 = F.leaky_relu(self.batch3_1(self.conv3_1(self.enc_layer2_pool)))
        self.enc_layer3 = F.leaky_relu(self.batch3_2(self.conv3_2(self.enc_layer3)))
        self.enc_layer3_pool = self.pool3(self.enc_layer3)

        self.bottleneck_layer = F.leaky_relu(self.bottleneck_batch(self.bottleneck_conv(self.enc_layer3_pool)))

        self.up3 = torch.cat((self.upsample_3(self.bottleneck_layer), self.enc_layer3), 1)
        self.up3 = F.leaky_relu(self.batch3_1(self.upconv3_1(self.up3)))
        self.up3 = F.leaky_relu(self.batch3_2(self.upconv3_2(self.up3)))

        self.up2 = torch.cat((self.upsample_2(self.up3), self.enc_layer2), 1)
        self.up2 = F.leaky_relu(self.batch2_1(self.upconv2_1(self.up2)))
        self.up2 = F.leaky_relu(self.batch2_2(self.upconv2_2(self.up2)))

        self.up1 = torch.cat((self.upsample_1(self.up2), self.enc_layer1), 1)
        self.up1 = F.leaky_relu(self.batch1_1(self.upconv1_1(self.up1)))
        self.up1 = F.leaky_relu(self.batch1_2(self.upconv1_2(self.up1)))

        self.out = self.conv_out(self.up1)

        return self.out
