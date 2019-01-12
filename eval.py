import os
import time
import cv2
import numpy as np
import torch
import torchvision
import torch.optim as optim

from model_resnet import Net
import dataset

net = Net().cuda()
net.load('./output/output-resnet_01111617/ultrasound.pth')
net.eval()

loader = dataset.create_eval_loader(batch_size = 10)

print('size: %d' % len(loader) )

IMAGE_HT = 224
IMAGE_WD = 224

dataiter = iter(loader)
images, labels = dataiter.next()
images = images.cuda()
output = net(images)

for idx in range(10):
    x, y, label = images[idx], output[idx], labels[idx]
    
    y = y.reshape(IMAGE_HT, IMAGE_WD).cpu().detach().numpy()
    label = label.reshape(IMAGE_HT, IMAGE_WD).cpu().detach().numpy()

    imgs = []

    # convert image to HSV for annotations
    orig = x.cpu().detach().numpy()
    orig = (orig + 1) * 127
    orig = orig.astype(np.uint8)
    orig = np.dstack((orig[0,:,:], orig[1,:,:], orig[2,:,:]))
    
    img = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    cv2.putText(img, 'Original', (5,20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0,0), 2)
    imgs.append(img)

    orig = cv2.cvtColor(orig, cv2.COLOR_RGB2HSV)

    # apply prediction and label markings

    img = np.copy(orig)
    h = img[:,:,0]
    s = img[:,:,1]
    h[label > .75] = 100 # BLUE
    s[label > .75] = 250
    cv2.putText(img, 'Label', (5,20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0,0), 2)
    imgs.append(cv2.cvtColor(img, cv2.COLOR_HSV2BGR))

    img = np.copy(orig)
    h = img[:,:,0]
    s = img[:,:,1]
    h[y > .75] = 50 # GREEN
    s[y > .75] = 250
    cv2.putText(img, 'Prediction', (5,20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0,0), 2)
    imgs.append(cv2.cvtColor(img, cv2.COLOR_HSV2BGR))
    
    img = np.copy(orig)
    h = img[:,:,0]
    s = img[:,:,1]
    h[y > .75] += 50 # GREEN
    s[y > .75] = 250
    h[label > .75] += 100 # BLUE
    s[label > .75] = 250
    cv2.putText(img, 'Combined', (5,20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0,0), 2)
    imgs.append(cv2.cvtColor(img, cv2.COLOR_HSV2BGR))

    final = np.hstack(imgs)

    cv2.imwrite('./output/resnet-%d.png'%idx, final)
