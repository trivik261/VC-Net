import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import torchvision.transforms.functional as F
import random
from utils.data_utils import decomposition_av, decomposition_av3, dataset_normalized

class MyDataset(Dataset):
    def __init__(self, dataset_name, list_file, channel=1, input_size = 512, is_train=True, transform=None):
        self.data_path_list = list_file
        self.imgs = []
        self.labels = []
        self.labels_v = []
        self.masks = []
        self.input_size = input_size
        self.channel = channel
        self.transform = transform
        self.is_train = is_train
        self.dataset_name = dataset_name
        for name_img in os.listdir(self.data_path_list):
            if self.channel == 3:
                img = Image.open(os.path.join(self.data_path_list, name_img)).convert('RGB')
            else:
                img = Image.open(os.path.join(self.data_path_list, name_img)).convert('L')
            self.imgs.append(img)

            label_path = os.path.join(self.data_path_list, name_img).replace('images', 'label')
            img_label = Image.open(label_path).convert('RGB')
            self.labels.append(img_label)

            mask_path = os.path.join(self.data_path_list, name_img).replace('images', 'mask')
            img_mask = Image.open(mask_path).convert('L')
            self.masks.append(img_mask)

            v_path = os.path.join(self.data_path_list, name_img).replace('images', 'vessel')
            img_v = Image.open(v_path).convert('L')
            self.labels_v.append(img_v)

    def __len__(self):
        return len(self.imgs)

    def add_img(self, tran, data):
        output = []
        for temp in data:
            output.append(tran(temp))
        output = tuple(output)
        return output

    def rotate_random_clip(self, data):
        output = []
        rotate_ = random.choice(range(90))
        w, h = data[0].size
        w_ = w - self.input_size
        h_ = h - self.input_size
        x1 = random.choice(range(w_))
        y1 = random.choice(range(h_))

        for img in data:
            img = img.rotate(rotate_)
            img = img.crop((x1, y1, x1+self.input_size, y1+self.input_size))
            output.append(img)
        output = tuple(output)

        return output

    def __getitem__(self, index):

        img = self.imgs[index]
        mask = self.masks[index]
        label1 = self.labels[index]
        label_v = self.labels_v[index]

        if self.is_train:
            if random.random() <0.5:
                (img, mask, label1, label_v) = self.add_img(transforms.RandomHorizontalFlip(p=1), (img, mask, label1, label_v))
            if random.random() < 0.5:
                (img, mask, label1, label_v) = self.add_img(transforms.RandomVerticalFlip(p=1), (img, mask, label1, label_v))
            (img, mask, label1, label_v) = self.rotate_random_clip((img, mask, label1, label_v))
            img = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)(img)

            label = decomposition_av(label1)

            (img, mask) = self.add_img(transforms.ToTensor(), (img, mask))
            mask[mask>0.5] = 1
            mask[mask<0.5] = 0

            v = np.copy(np.asarray(label_v))
            v[v > 0] = 1
            return img, torch.tensor(label), torch.tensor(v), torch.squeeze(mask)
        else:
            # im = dataset_normalized()
            w, h = img.size
            if self.dataset_name == 'DRIVE_AV':
                p = 32
            else:
                p = self.input_size
            w_ = w % p
            if w_ > 0:
                img = F.pad(img, ((p - w_) // 2, 0, p - w_ - (p - w_) // 2, 0))
                mask = F.pad(mask, ((p - w_) // 2, 0, p - w_ - (p - w_) // 2, 0))
                label1 = F.pad(label1, ((p - w_) // 2, 0, p - w_ - (p - w_) // 2, 0))
                label_v = F.pad(label_v, ((p - w_) // 2, 0, p - w_ - (p - w_) // 2, 0))

            h_ = h % p
            if h_ > 0:
                img = F.pad(img, (0, (p - h_) // 2, 0, p - h_ - (p - h_) // 2))
                mask = F.pad(mask, (0, (p - h_) // 2, 0, p - h_ - (p - h_) // 2))
                label1 = F.pad(label1, (0, (p - h_) // 2, 0, p - h_ - (p - h_) // 2))
                label_v = F.pad(label_v, (0, (p - h_) // 2, 0, p - h_ - (p - h_) // 2))

            if self.dataset_name == 'TR_AV':
                label = decomposition_av3(label1)
            else:
                label = decomposition_av(label1)

            (img, mask) = self.add_img(transforms.ToTensor(), (img, mask))
            mask[mask > 0.5] = 1
            mask[mask < 0.5] = 0

            v = np.copy(np.asarray(label_v))
            v[v > 0] = 1
            return img, torch.tensor(label), torch.tensor(v), torch.squeeze(mask)


