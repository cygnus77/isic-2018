import glob
import random
import csv
import cv2
import re
import numpy as np
import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms.functional import _get_inverse_affine_matrix

import math
import matplotlib.pyplot as plt

IMAGE_HT = 224
IMAGE_WD = 224

class RandomStreaks(object):
    def __init__(self):
        pass

    def __call__(self, item):
        img = item[0]
        for i in range(random.randint(0,4)):
            center = (random.randint(0, item[0].shape[0]), random.randint(0, item[0].shape[1]))
            axes = (random.randint(0, item[0].shape[0]), random.randint(0, item[0].shape[1]))
            angle = random.randint(0, 360)
            start = random.randint(0, 360)
            end = random.randint(45,180) + start
            thickness = random.choice([1,2])
            img = cv2.ellipse(img, center, axes, angle, start, end, (0,0,0,255), thickness)
        return (img, item[1])

class RandomAffine(object):
    def __init__(self):
        self.angle = random.randint(-180, 180)
        self.scale = 1./(1. + random.random())
        self.shear = random.randint(0,30)
        pass

    def __call__(self, item):
        img = item[0]
        tgt = item[1]

        tx = _get_inverse_affine_matrix((img.shape[0]//2, img.shape[1]//2), self.angle, (0,0), self.scale, self.shear)
        M = np.array(tx)
        M = np.reshape(M, (2,3))

        img = cv2.warpAffine(img, M, dsize=(img.shape[0],img.shape[1]))
        tgt = cv2.warpAffine(tgt, M, dsize=(tgt.shape[0],tgt.shape[1]))

        return img, tgt

class RandomSaturation(object):
    def __init__(self):
        self.s_shift = random.randint(0,50)
        pass
    def __call__(self, item):
        img = item[0]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        img[:,:,1] += self.s_shift
        img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
        return (img,item[1])

class LesionImageSizes(object):
    def __init__(self):
        self.table={}
        with open('./img_data.csv', 'r') as f:
            rdr = csv.reader(f)
            for row in rdr:
                imgno = re.match('.*ISIC_(\\d*)\\.jpg', row[0]).group(1)
                self.table[imgno] = (int(row[1]), int(row[2]))

    def getSize(self, fname):
        imgno = re.match('\\D*(\\d*)(_mask)?\\.png', fname).group(1)
        return self.table[imgno]

class LesionDataset(data.Dataset):
    highlighted_color = np.array([255, 255, 255])
    
    def __init__(self, mask_list, input_preprocessor, augment=False):
        super().__init__()
        self.y = mask_list
        self.x = [grp.group(1)+grp.group(2) for grp in [re.match(r'(.*)_mask(\.png)', x) for x in mask_list]]
        self.input_preprocessor = input_preprocessor
        self.augment = augment
        
    def imread(self, file_name):
        return cv2.cvtColor(cv2.imread(file_name), cv2.COLOR_BGR2RGB)
    
    def labelcvt(self, img):
        gt_bg = np.all(img == LesionDataset.highlighted_color, axis=2)
        gt_bg = np.expand_dims(gt_bg, 2)
        
        class1 = np.zeros(gt_bg.shape, dtype=np.float32)
        class1[gt_bg] = 1.
        return class1.reshape(-1, 1)

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        img = self.imread(self.x[idx])
        tgt = self.imread(self.y[idx])

        if self.augment:
            img,tgt = RandomStreaks()((img,tgt))
            img,tgt = RandomAffine()((img,tgt))
            img,tgt = RandomSaturation()((img,tgt))

        return self.input_preprocessor(img), self.labelcvt(tgt), self.y[idx]

    input_processor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

def create_loaders(val_percent = 20, batch_size = 10, augment=False):

    filelist = glob.glob('./ISIC/train-resized/*_mask.png')
    val_items = val_percent*len(filelist)//100

    random.shuffle(filelist)

    validation_list = filelist[0: val_items]
    val_dataset = LesionDataset(validation_list, LesionDataset.input_processor)
    val_loader = data.DataLoader(val_dataset, batch_size=batch_size, num_workers=2)

    train_list = filelist[val_items:]
    train_dataset = LesionDataset(train_list, LesionDataset.input_processor, augment=augment)
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, num_workers=2, shuffle=True)

    return (train_loader, val_loader)

def create_eval_loader(batch_size):
    filelist = glob.glob('./ISIC/train-resized/*_mask.png')
    random.shuffle(filelist)
    dataset = LesionDataset(filelist, LesionDataset.input_processor, augment=False)
    loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return loader

if __name__ == "__main__":

    sizeLookup = LesionImageSizes()

    images, labels, fnames = iter(create_eval_loader(10)).next()
    
    for i in range(10):
        print(fnames[i], sizeLookup.getSize(fnames[i]))

        img = images[i].cpu().detach().numpy()
        label = labels[i].cpu().detach().numpy().reshape(224,224)

        img = (img + 1) * 127
        img = img.astype(np.uint8)
        img = np.dstack((img[0,:,:], img[1,:,:], img[2,:,:]))

        label = (label + 1) * 127
        label = label.astype(np.uint8)

        fig = plt.figure()
        plt.subplot(1,2,1)
        plt.imshow(img)
        plt.subplot(1,2,2)
        plt.imshow(label)
        plt.show()
