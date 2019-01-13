import os
import time
import cv2
import numpy as np
import torch
import torchvision
import torch.optim as optim

from model_resnet import Net
import dataset

class Evaluator(object):

    def __init__(self, model_filepath, outputDir):
        self.outputDir = outputDir
        self.net = Net().cuda()
        self.net.load(model_filepath)
        self.net.eval()
        self.imagesizes = dataset.LesionImageSizes()
        if not os.path.exists(outputDir):
            os.mkdir(outputDir)

    def _makeAnnotatedImage(self, x, y, label):    
        y = y.reshape(dataset.IMAGE_HT, dataset.IMAGE_WD).cpu().detach().numpy()
        label = label.reshape(dataset.IMAGE_HT, dataset.IMAGE_WD).cpu().detach().numpy()

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
        return final

    def _saveAnnotatedOutput(self, x, y, label, idx):
        final = self._makeAnnotatedImage(x, y, label)
        cv2.imwrite(os.path.join(self.outputDir, '%d.png'%idx), final)

    def _saveMask(self, y, fname):
        y = y.reshape(dataset.IMAGE_HT, dataset.IMAGE_WD).cpu().detach().numpy()
        mask = np.zeros_like(y, dtype=np.uint8)
        mask[y>0.75] = 255
        size = self.imagesizes.getSize(fname)
        mask = cv2.resize(mask, size, cv2.INTER_CUBIC)
        cv2.imwrite(os.path.join(self.outputDir, os.path.split(fname)[1]), mask)

    def run(self, annotate=True, save_mask=True):
        
        loader = dataset.create_eval_loader(batch_size = 10)

        print('size: %d' % len(loader) )       

        for images, labels, fnames in loader:
            images = images.cuda()
            output = self.net(images)
            for idx in range(10):
                x, y, label, fname = images[idx], output[idx], labels[idx], fnames[idx]
                if annotate:
                    self._saveAnnotatedOutput(x, y, label, idx)
                if save_mask:
                    self._saveMask(y, fname)

    def sample(self):
        loader = dataset.create_eval_loader(batch_size = 10)
        images, labels, fnames = iter(loader).next()
        images = images.cuda()
        output = self.net(images)
        samples = []
        for idx in range(10):
            x, y, label, fname = images[idx], output[idx], labels[idx], fnames[idx]
            samples.append(self._makeAnnotatedImage(x, y, label))
        return samples


if __name__ == "__main__":
    evaluator = Evaluator('./output/resnet-01122310/lesions.pth', './output/test')
    evaluator.sample()
    #evaluator.run(annotate=False, save_mask=True)