import os
import cv2
import numpy as np
import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

class Unet(torch.nn.Module):
	def __init__(self, image_channels, hidden_size=16, n_classes=4):	
		super(Unet, self).__init__()
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		# Encoder
		self.conv1_1 = nn.Conv2d(in_channels = image_channels, out_channels = hidden_size, kernel_size = 3, stride = 1, padding = 1)
		self.batch1_1 = nn.BatchNorm2d(hidden_size, affine=True) 
		self.conv1_2 = nn.Conv2d(in_channels = hidden_size, out_channels = hidden_size, kernel_size = 3, stride = 1, padding = 1)
		self.batch1_2 = nn.BatchNorm2d(hidden_size, affine=True)
		self.pool1 = nn.MaxPool2d(3, 2, padding=1)

		self.conv2_1 = nn.Conv2d(in_channels = hidden_size, out_channels = hidden_size*2, kernel_size = 3, stride = 1, padding = 1)
		self.batch2_1 = nn.BatchNorm2d(hidden_size*2, affine=True)
		self.conv2_2 = nn.Conv2d(in_channels = hidden_size*2, out_channels = hidden_size*2, kernel_size = 3, stride = 1, padding = 1)
		self.batch2_2 = nn.BatchNorm2d(hidden_size*2, affine=True)
		self.pool2 = nn.MaxPool2d(3, 2, padding=1)

		self.conv3_1 = nn.Conv2d(in_channels = hidden_size*2, out_channels = hidden_size*4, kernel_size = 3, stride = 1, padding = 1)
		self.batch3_1 = nn.BatchNorm2d(hidden_size*4, affine=True)
		self.conv3_2 = nn.Conv2d(in_channels = hidden_size*4, out_channels = hidden_size*4, kernel_size = 3, stride = 1, padding = 1)
		self.batch3_2 = nn.BatchNorm2d(hidden_size*4, affine=True)
		self.pool3 = nn.MaxPool2d(3, 2, padding=1)

		# Bottleneck
		self.bottleneck_conv = nn.Conv2d(in_channels = hidden_size*4, out_channels = hidden_size*8, kernel_size = 3, stride = 1, padding = 1)
		self.bottleneck_batch = nn.BatchNorm2d(hidden_size*8, affine=True)

		# Decoder
		self.upsample_3 = nn.ConvTranspose2d(in_channels = hidden_size*8, out_channels = hidden_size*4, kernel_size = 2, stride = 2)
		self.upconv3_1 = nn.Conv2d(in_channels = hidden_size*8, out_channels = hidden_size*4, kernel_size = 3, stride = 1, padding = 1)
		self.upbatch3_1 = nn.BatchNorm2d(hidden_size*4, affine=True)
		self.upconv3_2 = nn.Conv2d(in_channels = hidden_size*4, out_channels = hidden_size*4, kernel_size = 3, stride = 1, padding = 1)
		self.upbatch3_2 = nn.BatchNorm2d(hidden_size*4, affine=True)

		self.upsample_2 = nn.ConvTranspose2d(in_channels = hidden_size*4, out_channels = hidden_size*2, kernel_size = 2, stride = 2)
		self.upconv2_1 = nn.Conv2d(in_channels = hidden_size*4, out_channels = hidden_size*2, kernel_size = 3, stride = 1, padding = 1)
		self.upbatch2_1 = nn.BatchNorm2d(hidden_size*2, affine=True)
		self.upconv2_2 = nn.Conv2d(in_channels = hidden_size*2, out_channels = hidden_size*2, kernel_size = 3, stride = 1, padding = 1)
		self.upbatch2_2 = nn.BatchNorm2d(hidden_size*2, affine=True)

		self.upsample_1 = nn.ConvTranspose2d(in_channels = hidden_size*2, out_channels = hidden_size, kernel_size = 2, stride = 2)
		self.upconv1_1 = nn.Conv2d(in_channels = hidden_size*2, out_channels = hidden_size, kernel_size = 3, stride = 1, padding = 1)
		self.upbatch1_1 = nn.BatchNorm2d(hidden_size, affine=True)
		self.upconv1_2 = nn.Conv2d(in_channels = hidden_size, out_channels = hidden_size, kernel_size = 3, stride = 1, padding = 1)
		self.upbatch1_2 = nn.BatchNorm2d(hidden_size, affine=True)

		# Final Layer
		self.conv_out = nn.Conv2d(in_channels = hidden_size, out_channels = n_classes, kernel_size = 1, stride = 1, padding = 0)

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

class DatasetClass(data.Dataset):
	def __init__(self, root=''):
		super(DatasetClass, self).__init__()
		self.img_files = glob.glob(os.path.join(root, 'image', '*.png'))
		self.mask_files = []
		for img_path in self.img_files:
			basename = os.path.basename(img_path)
			self.mask_files.append(os.path.join(root,'mask',basename[:-4]+'_mask.png'))			

	def __getitem__(self, index):
		img_path = self.img_files[index]
		mask_path = self.mask_files[index]
		data = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
		label = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
		return torch.from_numpy(data).float(), torch.from_numpy(label).float()

	def __len__(self):
		return len(self.img_files)

class Main:
	def __init__(self, train__dir='./data/train', test_dir='./data/test', epochs = 10):
		self.train_dir, self.test_dir = train__dir, test_dir
		self.epochs = epochs
		self.train_dataset_size = len(glob.glob(os.path.join(self.train_dir, 'image', "*.png")))
		self.train_dataloader = DatasetClass(self.train_dir)
		self.test_dataloader = DatasetClass(self.test_dir)
		self.model = Unet(image_channels=1, hidden_size=16)
		self.data_viz()	# Un comment to visualize data
		# self.model_test()	# Un comment to test forward pass of model
	   
	def data_viz(self):
		for i in range(self.train_dataset_size):
			image = cv2.imread(os.path.join(self.train_dir, 'image', 'cmr{}.png'.format(str(i + 1))), cv2.IMREAD_UNCHANGED)
			mask = cv2.imread(os.path.join(self.train_dir, 'mask', 'cmr{}_mask.png'.format(str(i + 1))), cv2.IMREAD_UNCHANGED)
			self.dummy_mask = np.zeros((image.shape[0], image.shape[1], len(np.unique(mask))))
			self.mask_prepare(image, mask)

	def mask_prepare(self, img, mask):
		for i in range(len(np.unique(mask))):
			self.dummy_mask[:, :, i][np.where(mask==i)] = 255
		out = np.hstack((img.astype(np.uint8), self.dummy_mask[:,:,0].astype(np.uint8), self.dummy_mask[:,:,1].astype(np.uint8), self.dummy_mask[:,:,2].astype(np.uint8), self.dummy_mask[:,:,3].astype(np.uint8)))
		cv2.imshow("IMG",out)
		k = cv2.waitKey()
		if k == ord('q'):
			exit()

	def train(self):
		for epoch in range(self.epochs):
			for batch_idx, inputs in enumerate(self.train_dataloader):
				pass

	def model_test(self):
		img = cv2.imread(os.path.join(self.train_dir, 'image', 'cmr1.png'), cv2.IMREAD_UNCHANGED)
		self.img_tensor = torch.from_numpy(np.expand_dims(np.expand_dims(img.astype(np.uint8), 0), 0)).float()
		out = self.model(self.img_tensor)
		print(out.shape)

if __name__ == '__main__':
	Main()
