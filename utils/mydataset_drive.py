import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import torchvision.transforms.functional as F
import random


class MyDataset(Dataset):
    def __init__(self, list_file, channel=1, input_size = 512, is_train=True, transform=None):
        self.data_path_list_merge = list_file
        self.imgs = []
        self.labels = []
        self.masks = []
        self.flag = []
        self.input_size = input_size
        self.channel = channel
        self.transform = transform
        self.is_train = is_train
        self.basename = []
        # print(os.listdir(self.data_path_list))
        for file in self.data_path_list_merge:
            self.data_path_list = file

            for name_img in os.listdir(self.data_path_list):
                if self.channel == 3:
                    img = Image.open(os.path.join(self.data_path_list, name_img)).convert('RGB')
                else:
                    img = Image.open(os.path.join(self.data_path_list, name_img)).convert('L')
                self.imgs.append(img)


                label_path = os.path.join(self.data_path_list, name_img).replace('image', 'av_label')
                label = Image.open(label_path).convert('L')
                self.labels.append(label)
                mask_path = os.path.join(self.data_path_list, name_img.replace('.png', '.gif')).replace('image', 'masked')
                mask = Image.open(mask_path).convert('L')
                self.masks.append(mask)
                self.flag.append(label_path)
                self.basename.append(os.path.basename(label_path))

    def __len__(self):
        return len(self.imgs)

    def add_img1(self, tran, img, mask):
        img = tran(img)
        mask = tran(mask)
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
        label = self.labels[index]
        basename = self.basename[index]
        # print(index, flag, '\n')
        if self.is_train:
            if random.random() <0.5:
                img, mask, label = self.add_img(transforms.RandomHorizontalFlip(p=1), img, mask, label)
            if random.random() < 0.5:
                img, mask, label = self.add_img(transforms.RandomVerticalFlip(p=1), img, mask, label)
            img, mask, label = self.rotate_random_clip(img, mask, label)
            # img, mask, label1 = self.add_img(transforms.Resize((384, 384)), img, mask, label1)
            img = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)(img)

            label1 = np.copy(np.asarray(label))
            label1[label1 == 29] = 1 #vein
            label1[label1 == 76] = 2 #artery
            label1[label1 == 105] = 2 #artery
            label1[label1 == 150] = 2 #cross point
            label1[label1 == 255] = 3 #unknow point

            img, mask = self.add_img1(transforms.ToTensor(), img, mask)
            mask1 = np.asarray(mask)
            mask1[mask1>0.5] = 1
            mask1[mask1<0.5] = 0

            label_v = np.copy(label)
            label_v[label_v > 0] = 1
            mask = torch.tensor(mask1, dtype=torch.long)
            label = torch.tensor(label1, dtype=torch.long)
            label_v = torch.tensor(label_v, dtype=torch.long)

            return img, mask, label, label_v,  basename


        else:
            # w, h = img.size
            # p = 32
            # w_ = w % p
            # if w_ > 0:
            #     img = F.pad(img, ((p - w_) // 2, 0, p - w_ - (p - w_) // 2, 0))
            #     mask = F.pad(mask, ((p - w_) // 2, 0, p - w_ - (p - w_) // 2, 0))
            #     label = F.pad(label, ((p - w_) // 2, 0, p - w_ - (p - w_) // 2, 0))
            # h_ = h % p
            # if h_ > 0:
            #     img = F.pad(img, (0, (p - h_) // 2, 0, p - h_ - (p - h_) // 2))
            #     mask = F.pad(mask, (0, (p - h_) // 2, 0, p - h_ - (p - h_) // 2))
            #     label = F.pad(label, (0, (p - h_) // 2, 0, p - h_ - (p - h_) // 2))
            img, mask, label = self.add_img(transforms.Resize((512, 512)), img, mask, label)

            label1 = np.copy(np.asarray(label))

            label1[label1 == 29] = 1  # vein
            label1[label1 == 76] = 2  # artery
            label1[label1 == 150] = 2  # cross point
            label1[label1 == 255] = 3  # cross point

            img, mask = self.add_img1(transforms.ToTensor(), img, mask)
            mask[mask > 0.5] = 1
            mask[mask < 0.5] = 0

            label_v = np.copy(label)
            label_v[label_v > 0] = 1
            return img, torch.squeeze(mask), torch.tensor(label1), torch.tensor(label_v), basename


# train_data = MyDataset(
#     ['/storage/wanghua/pc_node4/VA-Net/DRIVE_AV/training_TR/images',
#                                                # '/storage/wanghua/pc_node4/VA-Net/Data_TR/training_TR/images',],
#                                                # '/storage/wanghua/pc_node4/VA-Net/HRF_AV/training_TR/images',],
#                                                #  '/storage/wanghua/pc_node4/VA-Net/KAILUAN_AV/training_TR/images',],
#                                                 '/storage/wanghua/pc_node4/VA-Net/LES_AV/training_TR/images',],
#                        channel=3, is_train=True, transform=None)
# # test_data = MyDataset(['DRIVE_AV/test/images'], channel=channels, is_train=False, transform=None)
# train_loader = torch.utils.data.DataLoader(train_data,
#         batch_size=2, shuffle=True, num_workers=2)
# # test_loader = torch.utils.data.DataLoader(test_data,
# #         batch_size=1, shuffle=False, num_workers=2)
# for step, data in enumerate(train_loader):
#     # 获取图片和标签
#     image, mask, label_av_, label_v_ = data
#
#     print('\n')
#     a = 1