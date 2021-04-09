"""
 * @author:  Hua Wang
 * @date: 2020-11-02 19:39
 * @version 0.0
"""

""""
Copyright (c) 2018. All rights reserved.
Created by Resnick Xing on 2018/5/10
"""

import numpy as np
import cv2
from skimage import morphology, measure
import math
import torch
import matplotlib.pyplot as plt

def connectTable(image,min_size,connect):
    label_image = measure.label(image)
    dst = morphology.remove_small_objects(label_image, min_size=min_size, connectivity=connect)
    return dst,measure.regionprops(dst)

def countWhite(image): #统计二值图中白色区域面积
    return np.count_nonzero(image)

def imgResize(image,scale):
    dim = (int(image.shape[1] *scale), int(image.shape[0] * scale))
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_LINEAR )
    return resized

def postprocess(probResult,probImage):
    dst,regionprops=connectTable(probResult,3000,1)
    result=np.zeros_like(probResult)
    prob=np.zeros_like(probImage)
    candidates = []   #被选择区域集
    probResult=probResult.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    probResult = cv2.morphologyEx(probResult, cv2.MORPH_CLOSE,kernel)
    probResult = cv2.morphologyEx(probResult, cv2.MORPH_CLOSE, kernel)

    for region in regionprops:  # 循环得到每一个连通区域属性集
        minr, minc, maxr, maxc = region.bbox
        area = (maxr - minr) * (maxc - minc)   #候选区域面积  area of selected patch

        if math.fabs((maxr - minr) / (maxc - minc)) > 1.3 or math.fabs((maxr - minr) / (maxc - minc)) < 0.8 or area * 4/3.1415926 < countWhite(probResult[minr:maxr, minc:maxc]):
            #剔除细、长区域和太过夸张的内凹型、外凸形 delete area which too small or big or wide etc
            continue
    #筛选过的区域与已选择区域合
        candidates.append(region.bbox)
    select_minr=0
    select_maxr=0
    select_minc=0
    select_maxc=0
    w_h_ratio=0
    #从原图中切割选择的区域  cut selected patch from origin image
    for candi in range(len(candidates)):
        minr, minc, maxr, maxc = candidates[candi]
        if math.fabs(w_h_ratio-1.0)>math.fabs((maxr - minr) / (maxc - minc)-1.0):
            select_minr = minr
            select_maxr = maxr
            select_minc = minc
            select_maxc = maxc
    result[select_minr :select_maxr , select_minc :select_maxc] = probResult[select_minr :select_maxr , select_minc :select_maxc]
    prob[select_minr :select_maxr , select_minc :select_maxc] = probImage[select_minr :select_maxr , select_minc :select_maxc]

    if np.max(prob)==0:
        prob=probImage
    return result.astype(np.uint8),prob


def get_test_patches(img, patch_size=512, stride=256, rl=False):
    """
    将待分割图预处理后，分割成patch
    :param img: 待分割图
    :patch_size: patch大小
    :stride: stride大小，隔多少距离去一个patch
    :return:
    """
    # test_img_adjust=img_process(test_img,rl=rl)  #预处理
    # img = img.transpose(1, 3)

    test_imgs=paint_border(img, patch_size, stride)  #将图片补足到可被完美分割状态

    test_img_patch=extract_patches(test_imgs, patch_size, stride)  #依顺序分割patch

    # test_img_patch = test_img_patch.transpose(1, 3)

    return test_img_patch,img.shape[2],test_imgs.shape[2]#,test_img_adjust

def paint_border(imgs,patch_size, stride):
    """
    将图片补足到可被完美分割状态
    :param imgs:  预处理后的图片
    :patch_size: patch大小
    :stride: stride大小，隔多少距离去一个patch
    :return:
    """
    assert (len(imgs.shape) == 4)
    img_h = imgs.shape[2]  # height of the full image
    img_w = imgs.shape[3]  # width of the full image
    leftover_h = (img_h - patch_size) % stride  # leftover on the h dim
    leftover_w = (img_w - patch_size) % stride  # leftover on the w dim
    full_imgs=imgs  #设置成None时 一些stride情况下会报错，比如stride=1
    if (leftover_h != 0):  #change dimension of img_h
        tmp_imgs = torch.zeros((imgs.shape[0],imgs.shape[1],img_h+(stride-leftover_h),img_w))
        tmp_imgs[0:imgs.shape[0],0:imgs.shape[1],0:img_h,0:img_w] = imgs
        full_imgs = tmp_imgs
    if (leftover_w != 0):   #change dimension of img_w
        tmp_imgs = torch.zeros((full_imgs.shape[0],full_imgs.shape[1],
                                full_imgs.shape[2],img_w+(stride - leftover_w)))
        tmp_imgs[0:imgs.shape[0],0:full_imgs.shape[1],0:imgs.shape[2],0:img_w] =imgs
        full_imgs = tmp_imgs
    print("new full images shape: \n" +str(full_imgs.shape))
    return full_imgs

def extract_patches(full_imgs, patch_size, stride):
    """
    按顺序分割patch
    :param full_imgs: 补足后的图片
    :patch_size: patch大小
    :stride: stride大小，隔多少距离去一个patch
    :return: 分割后的patch
    """
    assert (len(full_imgs.shape)==4)  #4D arrays
    img_h = full_imgs.shape[2]  #height of the full image
    img_w = full_imgs.shape[3] #width of the full image

    assert ((img_h-patch_size)%stride==0 and (img_w-patch_size)%stride==0)
    N_patches_img = ((img_h-patch_size)//stride+1)*((img_w-patch_size)//stride+1)  #// --> division between integers
    N_patches_tot = N_patches_img*full_imgs.shape[0]

    patches = torch.empty((N_patches_tot,full_imgs.shape[1],patch_size,patch_size))
    iter_tot = 0   #iter over the total number of patches (N_patches)
    for i in range(full_imgs.shape[0]):  #loop over the full images
        for h in range((img_h-patch_size)//stride+1):
            for w in range((img_w-patch_size)//stride+1):
                patch = full_imgs[i,:,h*stride:(h*stride)+patch_size,w*stride:(w*stride)+patch_size]
                patches[iter_tot]=patch
                iter_tot +=1   #total
    assert (iter_tot==N_patches_tot)
    return patches

def pred_to_patches(pred,config):
    """
    将预测的向量 转换成patch形态
    :param pred: 预测结果
    :param config: 配置文件
    :return: Tensor [-1，patch_height,patch_width,seg_num+1]
    """
    assert (len(pred.shape)==3)  #3D array: (Npatches,height*width,2)

    pred_images = np.empty((pred.shape[0],pred.shape[1],config.seg_num+1))  #(Npatches,height*width)
    pred_images[:,:,0:config.seg_num+1]=pred[:,:,0:config.seg_num+1]
    pred_images = np.reshape(pred_images,(pred_images.shape[0],config.patch_height,config.patch_width,config.seg_num+1))
    return pred_images

def img_process(data,rl=False):
    """
    预处理图片
    :param data: 输入图片
    :param rl: 原始图片是否预处理过
    :return: 预处理结果
    """
    assert(len(data.shape)==4)
    data=data.transpose(0, 3, 1,2)
    if rl==False:#原始图片是否已经预处理过
        train_imgs=np.zeros(data.shape)
        for index in range(data.shape[1]):
            train_img=np.zeros([data.shape[0],1,data.shape[2],data.shape[3]])
            train_img[:,0,:,:]=data[:,index,:,:]
            train_img = dataset_normalized(train_img)   #归一化
            train_img = clahe_equalized(train_img)      #限制性直方图归一化
            train_img = adjust_gamma(train_img, 1.2)    #gamma校正
            train_img = train_img/255.  #reduce to 0-1 range
            train_imgs[:,index,:,:]=train_img[:,0,:,:]

    else:
        train_imgs = np.zeros(data.shape)
        for index in range(data.shape[1]):
            train_img = np.zeros([data.shape[0], 1, data.shape[2], data.shape[3]])
            train_img[:, 0, :, :] = data[:, index, :, :]
            train_img = dataset_normalized(train_img)
            train_imgs[:, index, :, :] = train_img[:, 0, :, :]/255.

    train_imgs=train_imgs.transpose(0, 2, 3, 1)
    return train_imgs



def recompone_overlap(preds,patch_size, stride,img_h,img_w):
    """
    将patch拼成原始图片
    :param preds: patch块
    :param config: 配置文件
    :param img_h:  原始图片 height
    :param img_w:  原始图片 width
    :return:  拼接成的图片
    """
    if len(preds.shape)!=4:
        preds = preds.unsqueeze(0)
    assert (len(preds.shape)==4)  #4D arrays

    patch_h = patch_size
    patch_w = patch_size
    N_patches_h = (img_h-patch_h)//stride+1
    N_patches_w = (img_w-patch_w)//stride+1
    N_patches_img = N_patches_h * N_patches_w
    print("N_patches_h: " +str(N_patches_h))
    print("N_patches_w: " +str(N_patches_w))
    print("N_patches_img: " +str(N_patches_img))
    #assert (preds.shape[0]%N_patches_img==0)
    N_full_imgs = preds.shape[0]//N_patches_img
    print("According to the dimension inserted, there are " +str(N_full_imgs) +" full images (of " +str(img_h)+"x" +str(img_w) +" each)")
    full_prob = torch.zeros((N_full_imgs,preds.shape[1],img_h,img_w))  #itialize to zero mega array with sum of Probabilities
    full_sum = torch.zeros((N_full_imgs,preds.shape[1],img_h,img_w))

    k = 0 #iterator over all the patches
    for i in range(N_full_imgs):
        for h in range((img_h-patch_h)//stride+1):
            for w in range((img_w-patch_w)//stride+1):
                full_prob[i,:,h*stride:(h*stride)+patch_h,w*stride:(w*stride)+patch_w]+=preds[k]
                full_sum[i,:,h*stride:(h*stride)+patch_h,w*stride:(w*stride)+patch_w]+=1
                k+=1

    assert(k==preds.shape[0])
    assert(torch.min(full_sum)>=1)  #at least one
    final_avg = full_prob/full_sum
    print('using avg')
    return final_avg



#============================================================
#========= PRE PROCESSING FUNCTIONS ========================#
#============================================================

#==== histogram equalization
def histo_equalized(imgs):
    imgs_equalized = np.empty(imgs.shape)
    imgs_equalized = cv2.equalizeHist(np.array(imgs, dtype = np.uint8))
    return imgs_equalized


# CLAHE (Contrast Limited Adaptive Histogram Equalization)
#adaptive histogram equalization is used. In this, image is divided into small blocks called "tiles" (tileSize is 8x8 by default in OpenCV). Then each of these blocks are histogram equalized as usual. So in a small area, histogram would confine to a small region (unless there is noise). If noise is there, it will be amplified. To avoid this, contrast limiting is applied. If any histogram bin is above the specified contrast limit (by default 40 in OpenCV), those pixels are clipped and distributed uniformly to other bins before applying histogram equalization. After equalization, to remove artifacts in tile borders, bilinear interpolation is applied
def clahe_equalized(imgs):
    assert (len(imgs.shape)==4)  #4D arrays
    assert (imgs.shape[1]==1)  #check the channel is 1
    #create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    imgs_equalized = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        imgs_equalized[i,0] = clahe.apply(np.array(imgs[i,0], dtype = np.uint8))
    return imgs_equalized


# ===== normalize over the dataset
def dataset_normalized(imgs):
    assert (len(imgs.shape)==4)  #4D arrays
    assert (imgs.shape[1]==1)  #check the channel is 1
    imgs_normalized = np.empty(imgs.shape)
    imgs_std = np.std(imgs)
    imgs_mean = np.mean(imgs)
    imgs_normalized = (imgs-imgs_mean)/imgs_std
    for i in range(imgs.shape[0]):
        imgs_normalized[i] = ((imgs_normalized[i] - np.min(imgs_normalized[i])) / (np.max(imgs_normalized[i])-np.min(imgs_normalized[i])))*255
    return imgs_normalized


def adjust_gamma(imgs, gamma=1.0):
    assert (len(imgs.shape)==4)  #4D arrays
    assert (imgs.shape[1]==1)  #check the channel is 1
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    new_imgs = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        new_imgs[i,0] = cv2.LUT(np.array(imgs[i,0], dtype = np.uint8), table)
    return new_imgs

def decomposition_av(label_av):
    # label_av: given PIL Image
    label_av = np.copy(np.asarray(label_av))
    label = np.zeros_like(label_av[...,0])
    label[label_av[:, :, 0] == 255] = 3
    label[label_av[:, :, 2] == 255] = 2
    label[label_av[:, :, 1] == 255] = 1
    return label
