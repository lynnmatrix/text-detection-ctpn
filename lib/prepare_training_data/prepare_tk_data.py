# -*- coding:utf-8 -*-
import math
import os
import shutil

from PIL import Image
import xml.etree.ElementTree as ET
import cv2 as cv
import numpy as np

MAX_SHORT_SIDE = 600
MAX_LONG_SIDE = 1200
STEP = 16

SOURCE_ROOT_DIR = '../../../demo/test_images'
TARGET_ROOT_DIR = '/home/morgan/tk'


def __convert_tk_to_voc():
    if os.path.exists(TARGET_ROOT_DIR):
        shutil.rmtree(TARGET_ROOT_DIR)
    os.mkdir(TARGET_ROOT_DIR)
    tk_subdir = os.path.join(TARGET_ROOT_DIR, 'VOCTK')
    os.mkdir(tk_subdir)

    image_dir = os.path.join(tk_subdir, 'JPEGImages')
    annotation_dir = os.path.join(tk_subdir, 'Annotations')
    os.mkdir(os.path.join(tk_subdir, 'ImageSets'))
    os.mkdir(image_dir)
    os.mkdir(annotation_dir)

    trainset_dir = os.path.join(tk_subdir, 'ImageSets', 'Main')
    os.mkdir(trainset_dir)

    ftrain = open(os.path.join(trainset_dir, 'train.txt'), 'w')
    ftext_train = open(os.path.join(trainset_dir, 'text_train.txt'), 'w')
    fdontcare_train = open(os.path.join(trainset_dir, 'dontcare_train.txt'), 'w')

    for dirname, _, filenames in os.walk(SOURCE_ROOT_DIR):
        for filename in filenames:
            stem, ext = os.path.splitext(filename)
            ext = ext.strip('.')
            if ext in ['png', 'jpg']:
                print('convert {} to jpg'.format(filename))
                im = Image.open(os.path.join(dirname, filename))
                rgb_im = im.convert('RGB')
                rgb_im.save(os.path.join(image_dir, stem+'.jpg'))
            elif ext == 'xml':
                print('prepare annotation for {}'.format(filename))
                tree = ET.parse(os.path.join(dirname, filename))
                for name in tree.iter('name'):
                    name.text = 'text'
                tree.write(os.path.join(annotation_dir, filename))
                ftrain.writelines(stem + '\n')
                ftext_train.writelines(stem + ' 1\n')
                fdontcare_train.writelines(stem + ' -1\n')
    ftrain.close()
    ftext_train.close()
    fdontcare_train.close()


def __convert_tk_to_yolo():
    if os.path.exists(TARGET_ROOT_DIR):
        shutil.rmtree(TARGET_ROOT_DIR)
    os.mkdir(TARGET_ROOT_DIR)
    image_dir = 're_image'
    label_dir = 'label_tmp'
    if os.path.exists(image_dir):
        shutil.rmtree(image_dir)
    os.makedirs(image_dir)

    if os.path.exists(label_dir):
        shutil.rmtree(label_dir)
    os.makedirs(label_dir)

    for dirname, _, filenames in os.walk(SOURCE_ROOT_DIR):
        for filename in filenames:
            stem, ext = os.path.splitext(filename)
            ext = ext.strip('.')
            if ext not in ['png', 'jpg']:
                continue

            annotation_file = os.path.join(dirname, stem + '.xml')
            if not os.path.exists(annotation_file):
                print('No annotation file found for {}'.format(filename))
                continue

            img_path = os.path.join(dirname, filename)
            print(img_path)
            img = cv.imread(img_path)
            img_size = img.shape
            im_size_min = np.min(img_size[0:2])
            im_size_max = np.max(img_size[0:2])

            im_scale = float(MAX_SHORT_SIDE) / float(im_size_min)
            if np.round(im_scale * im_size_max) > MAX_LONG_SIDE:
                im_scale = float(MAX_LONG_SIDE) / float(im_size_max)
            re_im = cv.resize(img, None, None, fx=im_scale, fy=im_scale, interpolation=cv.INTER_LINEAR)
            cv.imwrite(os.path.join(image_dir, stem) + '.jpg', re_im)

            annotation_file = os.path.join(dirname, stem+'.xml')
            tree = ET.parse(annotation_file)

            for object in tree.iter('object'):
                rondbox = object.find('robndbox')
                if rondbox is not None:
                    cx = float(rondbox.find('cx').text) * im_scale
                    cy = float(rondbox.find('cy').text) * im_scale
                    width = float(rondbox.find('w').text) * im_scale
                    height = float(rondbox.find('h').text) * im_scale
                    angle = float(rondbox.find('angle').text)
                    if abs(angle) > 0.01:
                        print('annotation in {} has object with angle {}, ignore'.format(stem, angle))
                        continue

                    xmin = math.floor(cx - width / 2)
                    ymin = math.floor(cy - height / 2)
                    xmax = math.ceil(cx + width / 2)
                    ymax = math.ceil(cy + height / 2)
                else:
                    bndbox = object.find('bndbox')
                    if bndbox is None:
                        print('neither bndbox or rondbox found in {} '.format(stem))
                        continue
                    xmin = math.floor(float(bndbox.find('xmin').text))
                    ymin = math.floor(float(bndbox.find('ymin').text))
                    xmax = math.ceil(float(bndbox.find('xmax').text))
                    ymax = math.ceil(float(bndbox.find('ymax').text))

                x_left = []
                x_right = []
                x_left.append(xmin)
                x_left_start = int(math.ceil(xmin / float(STEP)) * float(STEP))
                if x_left_start == xmin:
                    x_left_start = xmin + STEP
                for i in np.arange(x_left_start, xmax, STEP):
                    x_left.append(i)
                x_left = np.array(x_left)

                x_right.append(x_left_start - 1)
                for i in range(1, len(x_left) - 1):
                    x_right.append(x_left[i] + (STEP - 1))
                x_right.append(xmax)
                x_right = np.array(x_right)

                idx = np.where(x_left == x_right)
                x_left = np.delete(x_left, idx, axis=0)
                x_right = np.delete(x_right, idx, axis=0)
                with open(os.path.join(label_dir, stem) + '.txt', 'a') as f:
                    for i in range(len(x_left)):
                        if x_left[i] > x_right[i]:
                            print 'x_left > x_right'
                        f.writelines("text\t")
                        f.writelines(str(int(x_left[i])))
                        f.writelines("\t")
                        f.writelines(str(int(ymin)))
                        f.writelines("\t")
                        f.writelines(str(int(x_right[i])))
                        f.writelines("\t")
                        f.writelines(str(int(ymax)))
                        f.writelines("\n")


if __name__ == '__main__':
    __convert_tk_to_yolo()




