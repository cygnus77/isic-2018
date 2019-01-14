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

def task(id, args, fileQueue):

    if args.resize is not None:
        if not os.path.exists(args.out):
            os.mkdir(args.out)

    if args.scan:
        
        wr = None
        if args.filename is not None:
            fname = re.sub('\\.csv', '_%d.csv'%id, args.filename)
            f = open(fname, 'w')
            wr = csv.writer(f)

        while True:

            filename = fileQueue.get()
            if filename is None:
                break;

            rowdata = []

            img = cv2.imread(filename)
            imgno = re.match('.*ISIC_(\\d*)\\.jpg', filename).group(1)
            label = cv2.imread('./ISIC/labels/ISIC_{}_segmentation.png'.format(imgno))

            rowdata.append(imgno)
            rowdata.extend(img.shape)

            if args.resize is not None:
                resized_img = cv2.resize(img, dsize=(args.resize,args.resize), interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(os.path.join(args.out, '{}.png'.format(imgno)), resized_img)
                resized_label = cv2.resize(label, dsize=(args.resize,args.resize), interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(os.path.join(args.out, '{}_mask.png'.format(imgno)), resized_label)

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

            if args.resize is not None:
                # write out cropped,scaled image
                roi_img = img[top:bottom, left:right]
                resized_img = cv2.resize(roi_img, dsize=(args.resize,args.resize), interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(os.path.join(args.out, 'roi_{}.png'.format(imgno)), resized_img)

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
                    resized_mask = cv2.resize(roi_mask, dsize=(args.resize,args.resize), interpolation=cv2.INTER_CUBIC)
                    cv2.imwrite(os.path.join(args.out, 'roi_{}_mask_{}.png'.format(imgno, j)), resized_mask)

            if wr is not None:
                wr.writerow(rowdata)
        
        if args.filename is not None:
            f.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dataset analysis tool')
    parser.add_argument('-filename', action='store', default=None, help='filename of image comparison data')
    parser.add_argument('-scan', action='store_true', help='list image dimensions, resize training and label image for task1. crop ROI, resize to 224x224 for task 2')
    parser.add_argument('-resize', action='store', nargs='?', type=int, const=224, help='resize to specified size')
    parser.add_argument('-out', action='store', default='./ISIC/train-resized', help='location to output scaled images')
    parser.add_argument('-num_procs', action='store', type=int, default=8, help='number of child processes')

    args = parser.parse_args()

    filelist = [x for x in glob('./ISIC/train/ISIC_*.jpg')]
    numfiles = len(filelist)
    print('num files: %d' %numfiles)

    # start processes with shared queue
    fileQueue = mp.Queue()
    procs = [mp.Process(target=task, args=(x, args, fileQueue)) for x in range(args.num_procs)]
    for proc in procs:
        proc.start()

    # write items to queue
    for f in filelist:
        fileQueue.put(f)

    # write quit signal
    for proc in procs:
        fileQueue.put(None)

    # wait for threads to die
    for proc in procs:
        proc.join()

