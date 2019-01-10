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
from multiprocessing import Process, Queue

parser = argparse.ArgumentParser(description='Dataset analysis tool')
parser.add_argument('-filename', action='store', default=None, help='filename of image comparison data')
parser.add_argument('-scan', action='store_true', help='list image dimensions')
parser.add_argument('-resize', action='store', nargs='?', type=int, const=224, help='resize to specified size')
parser.add_argument('-out', action='store', default='./ISIC/train-resized', help='location to output scaled images')

parser.add_argument('-hist', action='store_true', help='show histogram of difference amounts')
parser.add_argument('-bucket', action='store', type=int, help='set bucket size', default=10000 )
parser.add_argument('-bucketmax', action='store', type=int, help='max bucket', default=1e6)
parser.add_argument('-threshold', action='store', type=int, default=100, help='set threshold for detecting duplicates')
parser.add_argument('-view', nargs='?', action='store', type=int, const=100, help='show identical images and masks. optional arg to show descrepancies over a certain threshold')
parser.add_argument('-resolve', action='store', default=None, help='write out commands to resolve duplicates')

args = parser.parse_args()

if args.scan:
    filelist = [x for x in glob('./ISIC/train/ISIC_*.jpg')]
    numfiles = len(filelist)
    print('num files: %d' %numfiles)

    if args.resize is not None:
        if not os.path.exists(args.out):
            os.mkdir(args.out)

    csv = None
    if args.filename is not None:
        f = open(args.filename, 'w')
        csv = csv.writer(f)

    for i in range(numfiles):
        print(i,end='\r')
        img = cv2.imread(filelist[i])
        imgno = re.match('.*ISIC_(\\d*)\\.jpg', filelist[i]).group(1)
        label = cv2.imread('./ISIC/labels/ISIC_{}_segmentation.png'.format(imgno))

        if csv is not None:
            csv.writerow([imgno] + list(img.shape) + list(label.shape) + [img.shape[1]/img.shape[0]])

        if args.resize is not None:
            img = cv2.resize(img, dsize=(args.resize,args.resize), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(os.path.join(args.out, '{}.png'.format(imgno)), img)
            label = cv2.resize(label, dsize=(args.resize,args.resize), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(os.path.join(args.out, '{}_mask.png'.format(imgno)), label)
    
    if args.filename is not None:
        f.close()

    sys.exit(0)        
        
    image_names = [p[p.rfind('/')+1:-4] for p in filelist]
    # load image data into a 3d numpy array
    image_data = np.array([cv2.imread(p)[:,:,0] for p in filelist])

    batch_size = 200
    print('writing csv')
    with open(args.filename, 'w') as csv:
        csv.write('img1,img2,diff\n')
        for j in range(len(filelist)):
            print(j)
            # load image into cuda as tensor
            img = torch.from_numpy(image_data[j,:,:]).float().cuda()
            img = torch.unsqueeze(img,0)

            # process batches
            for i in range(j+1,len(filelist),batch_size):
                try:
                    # load batch of images into cuda
                    batch = torch.from_numpy(image_data[i:i+batch_size,:,:]).float().cuda()
                    
                    # calculate differences
                    diff = torch.sum(torch.abs(batch - img), [1,2])

                    # write out results
                    for k in range(diff.shape[0]):
                        csv.write('%s,%s,%d\n' % (image_names[j], image_names[i+k], diff[k]))
                except:
                    print(i,j)
                    raise

        csv.close()
    sys.exit()

if args.hist:
    histo = {}
    count = 0
    with open(args.filename, 'r') as f:
        rdr = csv.reader(f)
        for row in rdr:
            try:
                diff = int(row[2])
                #print('%s\t%s\t%d' % (row[0], row[1], diff))
                bucket = diff // args.bucket

                if bucket < args.bucketmax:
                    if bucket in histo:
                        histo[bucket] += 1
                    else:
                        histo[bucket] = 1

                    count += 1
            except:
                pass
        f.close()

    buckets = list(histo.keys())
    buckets.sort()

    values = [histo[x] for x in buckets]

    plt.figure()
    plt.bar( buckets, values )
    plt.show()

dup_count = 0
mask_mismatch_count = 0
groups = []
with open(args.filename, 'r') as f:
    rdr = csv.reader(f)
    for row in rdr:
        try:
            diff = int(row[2])
        except:
            continue
        if diff < args.threshold:
            dup_count += 1
            img1_mask = cv2.imread('./data/train_orig/%s_mask.tif' % row[0])
            img2_mask = cv2.imread('./data/train_orig/%s_mask.tif' % row[1])

            mask_diff = np.sum(np.abs(img1_mask - img2_mask))
            if mask_diff > args.threshold:
                mask_mismatch_count += 1

            grp = None
            for g in groups:
                if (row[0] in g['imgs']) or (row[1] in g['imgs']):
                    grp = g
                    break
            if grp is None:
                grp = {}
                grp['imgs'] = set()
                grp['edges'] = []
                groups.append(grp)
            grp['imgs'].add(row[0])
            grp['imgs'].add(row[1])
            grp['edges'].append({
                'a':row[0],
                'b':row[1],
                'diff': diff,
                'mask_diff': mask_diff
            })

print("Dup count={}, Mask mismatch={}".format(dup_count, mask_mismatch_count))
for g in groups:
    print([x for x in g['imgs']])

    if args.view is not None or args.resolve is not None:
        edges = [x for x in g['edges'] if x['diff'] > args.view]
        if len(edges) > 0:
            fig = plt.figure()
            #fig.suptitle(' '.join( ["{} - {} = {} ; ".format(e['a'], e['b'], e['diff']) for e in edges] ))

            imgs = set()
            for e in edges:
                imgs.add(e['a'])
                imgs.add(e['b'])

            imgs = list(imgs)
            for idx, img_name in enumerate(imgs):
                plt.subplot(2,len(imgs),idx+1).set_title("{}. {}".format(idx+1, img_name))
                img = cv2.imread('./data/train_orig/%s.tif' % img_name)
                plt.imshow(img, cmap='gray')
                plt.axis('off')

                plt.subplot(2,len(imgs),len(imgs)+idx+1)
                img = cv2.imread('./data/train_orig/%s_mask.tif' % img_name)
                plt.imshow(img, cmap='gray')
                plt.axis('off')
            
            if args.resolve is None:
                plt.show()
            else:
                plt.show(block=False)
                inp = input('Enter image numbers that are bad: ')
                if len(inp) > 0:
                    imgnos = map(int, inp.split(','))
                    mno = input('Enter mask to use: ')
                    if len(mno) > 0:
                        mno = int(mno)
                        with open(args.resolve, 'a') as f:
                            for imgno in imgnos:
                                f.write("cp -f './data/train_orig/{}_mask.tif' './data/train_orig/{}_mask.tif'\n".format(imgs[mno-1], imgs[imgno-1]))
                            f.close()
