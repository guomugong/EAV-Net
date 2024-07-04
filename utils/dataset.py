import torch
import cv2
import os
import glob
from torch.utils.data import Dataset
import random
from PIL import Image
import numpy as np
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import copy

class FundusSeg_Loader(Dataset):
    def __init__(self, data_path, is_train, dataset_name):
        self.dataset_name = dataset_name
        self.data_path = data_path

        if self.dataset_name == "avdrive":
            self.imgs_path = sorted(glob.glob(os.path.join(data_path, 'image/*.tif')))
            self.labels_path = sorted(glob.glob(os.path.join(data_path, 'label/*.png')))
            self.labels_vessel_path = sorted(glob.glob(os.path.join(data_path, 'label_vessel/*.tif')))

        self.is_train = is_train

    def __getitem__(self, index):
        image_path = self.imgs_path[index]
        label_path = self.labels_path[index]
        label_vessel_path = self.labels_vessel_path[index]

        image = Image.open(image_path)
        label = Image.open(label_path)
        label_vessel = Image.open(label_vessel_path)
        # convert let 3 channel become 1 channel , L present 1 channel
        label = label.convert('L')
        label_vessel = label_vessel.convert('L')

        if self.is_train == 0: # TEST
            if self.dataset_name == "avdrive":
                image, label, label_vessel = self.padding_image(image, label, label_vessel, 592, 592)

        if self.is_train == 1:
            if np.random.random_sample() <= 0.5:
                image, label, label_vessel = self.randomRotation(image, label, label_vessel)

            if np.random.random_sample() <= 0.25:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                label = label.transpose(Image.FLIP_LEFT_RIGHT)
                label_vessel = label_vessel.transpose(Image.FLIP_LEFT_RIGHT)

            if np.random.random_sample() <= 0.25:
                image = image.transpose(Image.FLIP_TOP_BOTTOM)
                label = label.transpose(Image.FLIP_TOP_BOTTOM)
                label_vessel = label_vessel.transpose(Image.FLIP_TOP_BOTTOM)

            if np.random.random_sample() <= 1:
                crop_size = 256
                w = random.uniform(0, image.size[0]-crop_size)
                h = random.uniform(0, image.size[1]-crop_size)
                image = TF.crop(image, w, h, crop_size, crop_size)
                label = TF.crop(label, w, h, crop_size, crop_size)
                label_vessel = TF.crop(label_vessel, w, h, crop_size, crop_size)

            color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)
            image_color = color_jitter(image)


        if self.is_train == 0:
            image_color = image

        image = np.asarray(image)
        image_color = np.asarray(image_color)
        label = np.asarray(label)
        label_vessel = np.asarray(label_vessel)

        image = image.transpose(2, 0, 1)
        image_color = image_color.transpose(2, 0, 1)
        label = label.reshape(label.shape[0], label.shape[1])
        label = np.array(label)

        label_vessel = label_vessel.reshape(1, label_vessel.shape[0], label_vessel.shape[1])
        label_vessel = np.array(label_vessel)
        label_vessel = label_vessel / 255

        sp = image_path.split('/')
        filename = sp[len(sp)-1]
        filename = filename[0:len(filename)-4] # del .tif

        return image, image_color, label, label_vessel, filename

    def __len__(self):
        return len(self.imgs_path)

    def randomRotation(self, image, label, label_vessel, mode=Image.BICUBIC):
        random_angle = np.random.randint(1, 360)
        return image.rotate(random_angle, mode), label.rotate(random_angle, Image.NEAREST), label_vessel.rotate(random_angle, Image.NEAREST)

    def padding_image(self,image, label, label_vessel, pad_to_h, pad_to_w):
        new_image = Image.new('RGB', (pad_to_w, pad_to_h), (0, 0, 0))
        new_label = Image.new('P', (pad_to_w, pad_to_h), (0, 0, 0))
        new_label_vessel = Image.new('P', (pad_to_w, pad_to_h), (0, 0, 0))
        new_image.paste(image, (0, 0))
        new_label.paste(label, (0, 0))
        new_label_vessel.paste(label_vessel, (0, 0))
        return new_image, new_label, new_label_vessel
