import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim

from ncprojectUnetClass import Unet
from ncprojectDataSetClass import DatasetClass


class Main:
	def __init__(self, train__dir='./data/train', test_dir='./data/test', val_dir="./data/val", epochs=5,
				 learning_rate=1e-3, batch_size=16, num_workers=2, load_model_params=True, save_model_params=True,
				 saved_params_path="models/modelparams.pt", save_freq=5, dataset_debug=True):

		# initialize class properties from arguments
		self.train_dir, self.test_dir = train__dir, test_dir
		self.epochs, self.lr = epochs, learning_rate
		self.batch_size, self.num_workers = batch_size, num_workers
		self.val_dir = val_dir
		self.save_model_params = save_model_params
		self.saved_params_path = saved_params_path
		self.save_freq = save_freq

		# determine if GPU can be used and assign to 'device' class property
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		# set properties relating to dataset
		self.train_dataset_size = len(glob.glob(os.path.join(self.train_dir, 'image', "*.png")))
		self.train_dataset = DatasetClass(self.train_dir)
		self.test_dataset = DatasetClass(self.test_dir)

		# load data ready for training
		self.train_dataloader = DataLoader(self.train_dataset, self.batch_size, shuffle=True,
										   num_workers=self.num_workers, pin_memory=True, drop_last=True)

		# create the model
		self.model = Unet(image_channels=1, hidden_size=16).to(self.device)

		# load model parameters from file
		if load_model_params and os.path.isfile(saved_params_path):
			self.model.load_state_dict(torch.load(saved_params_path, map_location=self.device))

		# set loss function
		self.loss = nn.CrossEntropyLoss()

		# set optimiser
		self.optim = optim.Adam(self.model.parameters(), self.lr)

		if dataset_debug:
			self.data_viz()	# Un comment to visualize data
			self.dataset_properties() # Un comment to find properties of training data
		
		# carry out training
		self.train()

		self.model_test()  # Un comment to test forward pass of model

	def dataset_properties(self):
		"""Find out statistics about number of pixels in the dataset belonging to each class"""
		class_scores = {}
		mask = None
		for i in range(self.train_dataset_size):
			mask = cv2.imread(os.path.join(self.train_dir, 'mask', 'cmr{}_mask.png'.format(str(i + 1))),
								cv2.IMREAD_UNCHANGED)
			for idx in np.unique(mask):
				pix_count = np.where(mask==idx)
				if idx not in class_scores.keys():
					class_scores[idx] = pix_count[0].shape[0]	# Create new keys in dict
				else:
					class_scores[idx] += pix_count[0].shape[0]	# Update already existing key
		plot_list = []
		if mask is not None:
			for idx in class_scores.keys():
				class_scores[idx] /= (mask.shape[0] * mask.shape[1] * self.train_dataset_size)
				plot_list.append(class_scores[idx]*100)	# Convert to percentage	

		plt.bar(["Class " + str(i) for i in range(len(plot_list))], np.array(plot_list), width = 0.1)
		plt.xlim(-0.5, len(plot_list))
		plt.ylim(0, 100)
		plt.xticks(np.arange(0, len(plot_list), 1))
		for i in range(len(plot_list)):
			plt.annotate(str(round(plot_list[i], 2)) + " %", xy=(i-0.1, plot_list[i]+2))
		plt.show()

	def data_viz(self):
		"""Displays each training data image with its corresponding masks"""
		for i in range(self.train_dataset_size):
			image = cv2.imread(os.path.join(self.train_dir, 'image', 'cmr{}.png'.format(str(i + 1))),
							   cv2.IMREAD_UNCHANGED)
			mask = cv2.imread(os.path.join(self.train_dir, 'mask', 'cmr{}_mask.png'.format(str(i + 1))),
							  cv2.IMREAD_UNCHANGED)
			self.dummy_mask = np.zeros((image.shape[0], image.shape[1], len(np.unique(mask))))
			# returns True when visualization should continue
			if self.mask_prepare(image, mask):
				pass
			else:
				break

	def mask_prepare(self, img, mask):
		""" Format each features image to be white on the feature and black everywhere else.
			Then combine into one image and display."""
		#filter out all pixels that are not 255 for each feature
		for i in range(len(np.unique(mask))):
			self.dummy_mask[:, :, i][np.where(mask == i)] = 255

		imgs = [img.astype(np.uint8)]
		for i in range(len(np.unique(mask))):
			imgs.append(self.dummy_mask[:, :, i].astype(np.uint8))

		# combine each feature map into one image
		out = np.hstack(imgs)

		# display image
		cv2.imshow("Feature maps", out)

		# quit when q key is pressed
		k = cv2.waitKey()
		if k == ord('q'):
			return False
		return True

	def train(self):
		"""Carry out training on the model"""
		for epoch in range(self.epochs):
			epoch_loss = 0
			for batch_idx, (data, label) in enumerate(self.train_dataloader):
				data, label = data.to(self.device), label.to(self.device)

				# forward pass
				out = self.model(data)

				# loss calculation
				loss = self.loss(out, label)

				# back-propogation
				self.optim.zero_grad()
				loss.backward()
				self.optim.step()
				epoch_loss += loss
			print("EPOCH {}: ".format(epoch), epoch_loss / batch_idx)

			# save model parameters to file
			if ((epoch+1) % self.save_freq==0) and self.save_model_params:
				print("Saving model")
				torch.save(self.model.state_dict(), self.saved_params_path)

	def model_test(self):
		"""display performance on each of validation data"""
		for i in range(1, 21):
			iStr = "0" + str(i) if i < 10 else str(i)

			# load in base image
			img = cv2.imread(os.path.join(self.val_dir, 'image', 'cmr1{}.png'.format(iStr)), cv2.IMREAD_UNCHANGED)

			# load in correct mask
			img_mask = cv2.imread(os.path.join(self.val_dir, 'mask', 'cmr1{}_mask.png'.format(iStr)),
								  cv2.IMREAD_UNCHANGED)

			# convert base image to tensor
			self.img_tensor = torch.from_numpy(np.expand_dims(np.expand_dims(img.astype(np.uint8), 0), 0)).to(
				self.device).float()
			with torch.no_grad():
				# process through model
				out = self.model(self.img_tensor)
				out = F.softmax(out, 1).cpu().permute(0, 2, 3, 1).numpy()[0] * 255

				# stack processed images into one image
				out_img = np.hstack((img.astype(np.uint8),
									 out[:, :, 0].astype(np.uint8),  # background
									 out[:, :, 1].astype(np.uint8),  # right ventricle
									 out[:, :, 2].astype(np.uint8),  # myocardium
									 out[:, :, 3].astype(np.uint8)))  # left ventricle

				# process mask to produce mirror stacked image as above, with features in the same places
				out_actual = np.zeros((img_mask.shape[0], img_mask.shape[1], len(np.unique(img_mask))))
				for i in range(len(np.unique(img_mask))):
					out_actual[:, :, i][np.where(img_mask == i)] = 255
				out_actual_img = np.hstack((img.astype(np.uint8),
											out_actual[:, :, 0].astype(np.uint8),  # background
											out_actual[:, :, 1].astype(np.uint8),  # right ventricle
											out_actual[:, :, 2].astype(np.uint8),  # myocardium
											out_actual[:, :, 3].astype(np.uint8)))  # left ventricle

				# display both images
				cv2.imshow("MODEL OUTPUT", out_img)
				cv2.imshow("ACTUAL MASK", out_actual_img)
				k = cv2.waitKey()
				if k == ord('q'):
					exit()


if __name__ == '__main__':
	Main(load_model_params=False, save_model_params=True, dataset_debug=True)
