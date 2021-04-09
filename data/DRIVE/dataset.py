from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np
from skimage.io import imsave, imread
from skimage import io



class MyDataset(Dataset):
    def __init__(self, list_file, is_train=True):
        self.data_path_list = list_file
        self.imgs = []
        self.labels = []
        self.masks = []
        self.is_train = is_train

        for name_img in os.listdir(self.data_path_list):
            img = Image.open(os.path.join(self.data_path_list, name_img)).convert('RGB')
            self.imgs.append(img)

            if is_train:
                label_path = os.path.join(self.data_path_list, name_img.replace('training', 'manual1')
                                          .replace('.png', '.bmp')).replace('images', '1st_manual')
                mask_path = os.path.join(self.data_path_list, name_img.replace('.png', '_mask.gif'))\
                    .replace('images', 'mask')
            else:
                label_path = os.path.join(self.data_path_list,name_img.replace('test', 'manual1')
                                          .replace('.png', '.gif')).replace('images','1st_manual')

                mask_path = os.path.join(self.data_path_list, name_img.replace('.png', '_mask.gif'))\
                    .replace('images', 'mask')

            img_label = Image.open(label_path).convert('L')
            img_mask = Image.open(mask_path).convert('L')
            self.labels.append(img_label)
            self.masks.append(img_mask)


    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):

        img = self.imgs[index]
        mask = self.masks[index]
        label = self.labels[index]
        label = np.copy(np.asarray(label))

        if self.is_train:
            label[label == 29] = 2
            label[label == 28] = 2
            label[label == 30] = 2
            label[label == 31] = 2
            label[label == 32] = 2
            label[label == 33] = 2

            label[label == 76] = 3
            label[label == 91] = 3
            label[label == 90] = 3
            label[label == 108] = 3

            label[label == 255] = 1
        else:
            label[label == 76] = 254
            label[label == 29] = 253
            label[label == 149] = 252
            label[label == 225] = 251
            label[label < 251] = 0

            label[label == 255] = 1
            label[label == 254] = 3
            label[label == 253] = 2

            label[label == 252] = 1
            label[label == 251] = 1
        return img, label, mask

def restruction_av(data):
    mat1 = np.zeros(data.shape)
    mat2 = np.zeros(data.shape)
    mat3 = np.zeros(data.shape)
    # a = ll==1
    mat1[data == 3] = 255
    mat3[data == 2] = 255
    mat2[data == 1] = 255

    mat = np.stack([mat1, mat2, mat3], -1)


    return mat/255