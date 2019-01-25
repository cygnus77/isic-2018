import os, sys
from glob import glob
import csv
import numpy as np
import re
import cv2
import torch
import matplotlib.pyplot as plt
import threading
import concurrent.futures as futures
import itertools
import argparse
import multiprocessing as mp

class Task():
    def __init__(self, args):
        self.args = args
    
    def start(self):
        filelist = self.getFileList()

        # start processes with shared queue
        self.fileQueue = mp.Queue()
        procs = [mp.Process(target=Task.run, args=[self, x]) for x in range(self.args.num_procs)]
        for proc in procs:
            proc.start()

        # write items to queue
        for f in filelist:
            self.fileQueue.put(f)

        # write quit signal
        for proc in procs:
            self.fileQueue.put(None)

        # wait for threads to die
        for proc in procs:
            proc.join()

        # merge outputs
        if self.args.filename:
            with open(args.filename, 'w') as fout:
                for id in range(self.args.num_procs):
                    fname = re.sub('\\.csv', '_%d.csv'%id, self.args.filename)
                    with open(fname, 'r') as fin:
                        fout.write(fin.read())
                    os.unlink(fname)



    def run(self, id):
        if self.args.resize is not None:
            if not os.path.exists(self.args.out):
                os.mkdir(self.args.out)

        wr = None
        if self.args.filename is not None:
            fname = re.sub('\\.csv', '_%d.csv'%id, self.args.filename)
            f = open(fname, 'w')
            wr = csv.writer(f)

        while True:
            filename = self.fileQueue.get()
            if filename is None:
                break;

            rowdata = self.processFile(filename)

            if wr is not None:
                wr.writerow(rowdata)
        
        if self.args.filename is not None:
            f.close()

    def processFile(self, filename):
        return []

    def getFileList(self):
        return []

# Scan: list image dimensions, resize training and label image for task1. crop ROI, resize to 224x224 for task 2
class ScanTask(Task):
    def __init__(self, args):
        super().__init__(args)

    def getFileList(self):
        return glob('./ISIC/train/ISIC_*.jpg')

    def processFile(self, filename):
        rowdata = []

        img = cv2.imread(filename)
        imgno = re.match('.*ISIC_(\\d*)\\.jpg', filename).group(1)
        label = cv2.imread('./ISIC/labels/ISIC_{}_segmentation.png'.format(imgno))

        rowdata.append(imgno)
        rowdata.extend(img.shape)

        if self.args.resize is not None:
            resized_img = cv2.resize(img, dsize=(self.args.resize, self.args.resize), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(os.path.join(self.args.out, '{}.png'.format(imgno)), resized_img)
            resized_label = cv2.resize(label, dsize=(self.args.resize, self.args.resize), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(os.path.join(self.args.out, '{}_mask.png'.format(imgno)), resized_label)

        # calculate ROI
        top = 0
        left = 0
        bottom = label.shape[0]
        right = label.shape[1]

        label_binary = label[:,:,0] == 255
        mrows = np.argwhere(np.any(label_binary, axis=1))
        mcols = np.argwhere(np.any(label_binary, axis=0))
        if len(mrows) > 0 and len(mcols) > 0:
            top = np.min(mrows)
            left = np.min(mcols)
            bottom = np.max(mrows)
            right = np.max(mcols)

        rowdata.extend([left, top, right, bottom])

        if self.args.resize is not None:
            # write out cropped,scaled image
            roi_img = img[top:bottom, left:right]
            resized_img = cv2.resize(roi_img, dsize=(self.args.resize, self.args.resize), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(os.path.join(self.args.out, 'roi_{}.png'.format(imgno)), resized_img)

            # attribute masks corresponding to image
            attr_mask_files = [
                './ISIC/task2-labels/ISIC_{}_attribute_globules.png'.format(imgno),
                './ISIC/task2-labels/ISIC_{}_attribute_milia_like_cyst.png'.format(imgno),
                './ISIC/task2-labels/ISIC_{}_attribute_negative_network.png'.format(imgno),
                './ISIC/task2-labels/ISIC_{}_attribute_pigment_network.png'.format(imgno),
                './ISIC/task2-labels/ISIC_{}_attribute_streaks.png'.format(imgno)
            ]

            for j, attr_mask_file in enumerate(attr_mask_files):
                # write out cropped, scaled images
                attr_mask = cv2.imread(attr_mask_file)
                roi_mask = attr_mask[top:bottom, left:right]
                resized_mask = cv2.resize(roi_mask, dsize=(self.args.resize, self.args.resize), interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(os.path.join(self.args.out, 'roi_{}_mask_{}.png'.format(imgno, j)), resized_mask)

        return rowdata

# Count: count pixels in mask images
class CountTask(Task):
    def __init__(self, args):
        super().__init__(args)

    def getFileList(self):
        return glob('./ISIC/train-resized/*mask*.png')

    def processFile(self, filename):
        m = re.search('.*?(\d+)_mask(_(\d+))?', filename)
        if m:
            img = cv2.imread(filename)
            total = np.size(img)
            ones = np.count_nonzero(img)
            imgno = m.groups()[0]
            classno = m.groups()[2] if m.groups()[2] is not None else -1
            return [imgno, classno, total, ones]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dataset analysis tool')
    parser.add_argument('function', action='store', help='Scan: list image dimensions, resize training and label image for task1. crop ROI, resize to 224x224 for task 2\nCount: count pixels in mask images')
    parser.add_argument('-filename', action='store', default=None, help='filename of image comparison data')
    parser.add_argument('-resize', action='store', nargs='?', type=int, const=224, help='resize to specified size')
    parser.add_argument('-out', action='store', default='./ISIC/train-resized', help='location to output scaled images')
    parser.add_argument('-num_procs', action='store', type=int, default=8, help='number of child processes')

    args = parser.parse_args()

    if args.function == 'scan':
        ScanTask(args).start()
    
    elif args.function == 'count':
        CountTask(args).start()

