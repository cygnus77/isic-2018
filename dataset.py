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
        return [img] + item[1:]

class RandomAffine(object):
    def __init__(self):
        self.angle = random.randint(-180, 180)
        self.scale = 1./(1. + random.random())
        self.shear = 0 #random.randint(0,30)
        pass

    def __call__(self, item):
        img = item[0]

        tx = _get_inverse_affine_matrix((img.shape[0]//2, img.shape[1]//2), self.angle, (0,0), self.scale, self.shear)
        M = np.array(tx)
        M = np.reshape(M, (2,3))

        return [cv2.warpAffine(x, M, dsize=(img.shape[0], img.shape[1]))  for x in item]

class RandomSaturation(object):
    def __init__(self):
        self.s_shift = random.randint(0,50)
        pass
    def __call__(self, item):
        img = item[0]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        img[:,:,1] += self.s_shift
        img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
        return [img]+ item[1:]

class LesionDataset(data.Dataset):

    highlighted_color = np.array([255, 255, 255])

    input_processor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    def __init__(self, x, y, imgnos, input_preprocessor, augment=False):
        super().__init__()
        self.imgnos = imgnos
        self.y = y
        self.x = x # [grp.group(1)+grp.group(2) for grp in [re.match(r'(.*)_mask(\.png)', x) for x in mask_list]]
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
        item = [self.imread(self.x[idx])] + [self.imread(y) for y in self.y[idx]]
 
        if self.augment:
            #item = RandomStreaks()(item)
            item = RandomAffine()(item)
            #item = RandomSaturation()(item)

        x = self.input_preprocessor(item[0])
        if len(item) > 2:
            y = np.dstack([self.labelcvt(tgt) for tgt in item[1:]]).squeeze()
        else:
            y = self.labelcvt(item[1])

        return x, y, self.imgnos[idx]


class LesionData(object):
    def __init__(self):
        self.table={}
        with open('./img_data.csv', 'r') as f:
            rdr = csv.reader(f)
            for row in rdr:
                self.table[row[0]] = (
                    ( int(row[1]), int(row[2]), int(row[3])), # shape
                    ( int(row[4]), int(row[5]), int(row[6]), int(row[7]) ) # ROI
                )

    def getShape(self, imgno):
        return self.table[imgno][0]

    def getROI(self, imgno):
        return self.table[imgno][1]
    
    def getImgNos(self):
        return list(self.table.keys())

    def getTask1TrainingDataLoaders(self, val_percent = 20, batch_size = 10, augment = False):
        imgnos = self.getImgNos()
        random.shuffle(imgnos)

        numval = val_percent*len(imgnos)//100

        val_imgnos = imgnos[0: numval]
        val_x = ['./ISIC/train-resized/{}.png'.format(n) for n in val_imgnos]
        val_y = [['./ISIC/train-resized/{}_mask.png'.format(n)] for n in val_imgnos]
        val_dataset = LesionDataset(val_x, val_y, val_imgnos, LesionDataset.input_processor, augment=False)

        train_imgnos = imgnos[numval:]
        train_x = ['./ISIC/train-resized/{}.png'.format(n) for n in train_imgnos]
        train_y = [['./ISIC/train-resized/{}_mask.png'.format(n)] for n in train_imgnos]
        train_dataset = LesionDataset(train_x, train_y, train_imgnos, LesionDataset.input_processor, augment=augment)

        return (data.DataLoader(train_dataset, batch_size=batch_size, num_workers=2, shuffle=True),
            data.DataLoader(val_dataset, batch_size=batch_size, num_workers=2))

    def getTask1EvalDataLoader(self, batch_size):
        imgnos = self.getImgNos()
        random.shuffle(imgnos)

        eval_x = ['./ISIC/train-resized/{}.png'.format(n) for n in imgnos]
        eval_y = [['./ISIC/train-resized/{}_mask.png'.format(n)] for n in imgnos]
        eval_dataset = LesionDataset(eval_x, eval_y, imgnos, LesionDataset.input_processor, augment=False)
        loader = data.DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
        return loader

    def getTask2TrainingDataLoaders(self, val_percent = 20, batch_size = 10, augment = False):
        imgnos = self.getImgNos()
        random.shuffle(imgnos)

        numval = val_percent*len(imgnos)//100

        val_imgnos = imgnos[0: numval]
        val_x = ['./ISIC/train-resized/roi_{}.png'.format(n) for n in val_imgnos]
        val_y = [['./ISIC/train-resized/roi_{}_mask_{}.png'.format(n,j) for j in range(5)] for n in val_imgnos]
        val_dataset = LesionDataset(val_x, val_y, val_imgnos, LesionDataset.input_processor, augment=False)

        train_imgnos = imgnos[numval:]
        train_x = ['./ISIC/train-resized/roi_{}.png'.format(n) for n in train_imgnos]
        train_y = [['./ISIC/train-resized/roi_{}_mask_{}.png'.format(n,j) for j in range(5)] for n in train_imgnos]
        train_dataset = LesionDataset(train_x, train_y, train_imgnos, LesionDataset.input_processor, augment=augment)

        return (data.DataLoader(train_dataset, batch_size=batch_size, num_workers=2, shuffle=True),
            data.DataLoader(val_dataset, batch_size=batch_size, num_workers=2))

    def getTask2EvalDataLoader(self, batch_size):
        imgnos = self.getImgNos()
        random.shuffle(imgnos)
        
        eval_x = ['./ISIC/train-resized/roi_{}.png'.format(n) for n in imgnos]
        eval_y = [['./ISIC/train-resized/roi_{}_mask_{}.png'.format(n,j) for j in range(5)] for n in imgnos]
        eval_dataset = LesionDataset(eval_x, eval_y, imgnos, LesionDataset.input_processor, augment=False)
        loader = data.DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
        return loader

if __name__ == "__main__":

    lesionData = LesionData()

    images, labels, fnames = iter(lesionData.getTask1EvalDataLoader(10)).next()
    
    for i in range(10):
        print(fnames[i], lesionData.getShape(fnames[i]))

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
