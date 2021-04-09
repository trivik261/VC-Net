import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import torchvision.transforms.functional as F
import random
from utils.data_utils import decomposition_av

class MyDataset(Dataset):
    def __init__(self, list_file, channel=1, input_size = 512, is_train=True, transform=None):
        self.data_path_list = list_file
        self.imgs = []
        self.labels = []
        self.labels_v = []
        self.masks = []
        self.input_size = input_size
        self.channel = channel
        self.transform = transform
        self.is_train = is_train
        for name_img in os.listdir(self.data_path_list):
            if self.channel == 3:
                img = Image.open(os.path.join(self.data_path_list, name_img)).convert('RGB')
            else:
                img = Image.open(os.path.join(self.data_path_list, name_img)).convert('L')
            self.imgs.append(img)

            label_path = os.path.join(self.data_path_list, name_img).replace('images', 'label')
            mask_path = os.path.join(self.data_path_list, name_img).replace('images', 'mask')
            img_label = Image.open(label_path).convert('RGB')
            img_mask = Image.open(mask_path).convert('L')
            self.labels.append(img_label)
            self.masks.append(img_mask)


    def __len__(self):
        return len(self.imgs)

    def add_img1(self, tran, img, mask):
        img = tran(img)
        mask = tran(mask)
        # label = tran(label)
        return img,mask#,label

    def add_img(self, tran, img, mask,label):
        img = tran(img)
        mask = tran(mask)
        label = tran(label)
        return img,mask,label

    def rotate_random_clip(self, img, mask, label):
        rotate_ = random.choice(range(90))
        img = img.rotate(rotate_)
        mask = mask.rotate(rotate_)
        label = label.rotate(rotate_)
        w, h = img.size
        w_ = w - self.input_size
        h_ = h - self.input_size

        x1 = random.choice(range(w_))
        y1 = random.choice(range(h_))
        img = img.crop((x1, y1, x1+self.input_size, y1+self.input_size))
        mask = mask.crop((x1, y1, x1+self.input_size, y1+self.input_size))
        label = label.crop((x1, y1, x1+self.input_size, y1+self.input_size))

        return img, mask, label

    def __getitem__(self, index):

        img = self.imgs[index]
        mask = self.masks[index]
        label1 = self.labels[index]

        if self.is_train:
            if random.random() <0.5:
                img, mask, label1 = self.add_img(transforms.RandomHorizontalFlip(p=1), img, mask, label1)
            if random.random() < 0.5:
                img, mask, label1 = self.add_img(transforms.RandomVerticalFlip(p=1), img, mask, label1)
            img, mask, label1 = self.rotate_random_clip(img, mask, label1)
            img = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)(img)

            label = decomposition_av(label1)

            img, mask = self.add_img1(transforms.ToTensor(), img, mask)
            mask[mask>0.5] = 1
            mask[mask<0.5] = 0

            label_v = np.copy(label)
            label_v[label_v > 0] = 1

            return img, torch.tensor(label), torch.tensor(label_v), torch.squeeze(mask)
        else:
            w, h = img.size
            p = 32
            w_ = w % p
            if w_ > 0:
                img = F.pad(img, ((p - w_) // 2, 0, p - w_ - (p - w_) // 2, 0))
                mask = F.pad(mask, ((p - w_) // 2, 0, p - w_ - (p - w_) // 2, 0))
                label1 = F.pad(label1, ((p - w_) // 2, 0, p - w_ - (p - w_) // 2, 0))

            h_ = h % p
            if h_ > 0:
                img = F.pad(img, (0, (p - h_) // 2, 0, p - h_ - (p - h_) // 2))
                mask = F.pad(mask, (0, (p - h_) // 2, 0, p - h_ - (p - h_) // 2))
                label1 = F.pad(label1, (0, (p - h_) // 2, 0, p - h_ - (p - h_) // 2))

            label = decomposition_av(label1)

            img, mask = self.add_img1(transforms.ToTensor(), img, mask)
            mask[mask > 0.5] = 1
            mask[mask < 0.5] = 0

            label1_v = np.copy(np.asarray(label))
            label1_v[label1_v > 0] = 1
            return img, torch.tensor(label), torch.tensor(label1_v), torch.squeeze(mask)


