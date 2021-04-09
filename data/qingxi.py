import os
import numpy as np
from skimage.io import imsave, imread
from skimage import io
import glob
_join = os.path.join

from data.DRIVE.dataset import MyDataset, restruction_av
from experiments.config import process_config

cfg = process_config('/data/wanghua/VC_Net/experiments/drive_av/standard.json')

train_data = MyDataset(cfg.train_data_path[0], is_train=True)
test_data = MyDataset(cfg.test_data_path[0], is_train=False)

if not os.path.exists(cfg.train_data_path1[0]):
    os.makedirs(_join(cfg.train_data_path1[0], 'images'))
    os.makedirs(_join(cfg.train_data_path1[0], 'mask'))
    os.makedirs(_join(cfg.train_data_path1[0], 'label'))
for step, data in enumerate(train_data):
    d, label, mask = data
    imsave(_join(cfg.train_data_path1[0], 'images/%s.png' % step), np.asarray(d))
    imsave(_join(cfg.train_data_path1[0], 'mask/%s.png' % step), np.asarray(mask))
    imsave(_join(cfg.train_data_path1[0], 'label/%s.png' % step), restruction_av(label))

if not os.path.exists(cfg.test_data_path1[0]):
    os.makedirs(_join(cfg.test_data_path1[0], 'images'))
    os.makedirs(_join(cfg.test_data_path1[0], 'mask'))
    os.makedirs(_join(cfg.test_data_path1[0], 'label'))
for step, data in enumerate(test_data):
    d, label, mask = data
    imsave(_join(cfg.test_data_path1[0], 'images/%s.png' % step), np.asarray(d))
    imsave(_join(cfg.test_data_path1[0], 'mask/%s.png' % step), np.asarray(mask))
    imsave(_join(cfg.test_data_path1[0], 'label/%s.png' % step), restruction_av(label))


    a = 1

a = 1
