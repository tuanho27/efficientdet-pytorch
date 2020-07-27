""" COCO dataset (quick and dirty)

Hacked together by Ross Wightman
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data

import os
import torch
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
import random
import cv2
debug = False


class CocoDetection(data.Dataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.
    Args:
        root (string): Root directory where images are downloaded to.
        ann_file (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``

    """

    def __init__(self, root, ann_file, transform=None):
        super(CocoDetection, self).__init__()
        if isinstance(root, torch._six.string_classes):
            root = os.path.expanduser(root)
        self.root = root
        self.transform = transform
        self.yxyx = False   # expected for TF model, most PT are xyxy
        self.include_masks = False
        self.include_bboxes_ignore = False
        self.has_annotations = True #'image_info' not in ann_file
        self.coco = None
        self.cat_ids = []
        self.cat_to_label = dict()
        self.img_ids = []
        self.img_ids_invalid = []
        self.img_infos = []
        self._load_annotations(ann_file)

    def _load_annotations(self, ann_file):
        assert self.coco is None
        self.coco = COCO(ann_file)
        self.cat_ids = self.coco.getCatIds()
        # img_ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
        self.img_ids = self.coco.getImgIds()
        for i in self.img_ids:
            info = self.coco.loadImgs([i])[0]
            self.img_infos.append(info)
        # for img_id in sorted(self.coco.imgs.keys()):
            # info = self.coco.loadImgs([img_id])[0]
            # valid_annotation = not self.has_annotations or img_id in img_ids_with_ann
            # if valid_annotation and min(info['width'], info['height']) >= 32:
            #     self.img_ids.append(img_id)
            #     self.img_infos.append(info)
            # else:
            #     self.img_ids_invalid.append(img_id)

    def _parse_img_ann(self, img_id, img_info, mix=False):
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        ann_info = self.coco.loadAnns(ann_ids)
        bboxes = []
        bboxes_ignore = []
        cls = []

        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            if self.include_masks and ann['area'] <= 0:
                continue
            if w < 1 or h < 1:
                continue

            # To subtract 1 or not, TF doesn't appear to do this so will keep it out for now.
            if self.yxyx and not mix:
                #bbox = [y1, x1, y1 + h - 1, x1 + w - 1]
                bbox = [y1, x1, y1 + h, x1 + w]
            else:
                #bbox = [x1, y1, x1 + w - 1, y1 + h - 1]
                bbox = [x1, y1, x1 + w, y1 + h]

            if ann.get('iscrowd', False):
                if self.include_bboxes_ignore:
                    bboxes_ignore.append(bbox)
            else:
                bboxes.append(bbox)
                cls.append(self.cat_to_label[ann['category_id']] if self.cat_to_label else ann['category_id'])

        if bboxes:
            bboxes = np.array(bboxes, dtype=np.float32)
            cls = np.array(cls, dtype=np.int64)
        else:
            bboxes = np.zeros((0, 4), dtype=np.float32)
            cls = np.array([], dtype=np.int64)

        if self.include_bboxes_ignore:
            if bboxes_ignore:
                bboxes_ignore = np.array(bboxes_ignore, dtype=np.float32)
            else:
                bboxes_ignore = np.zeros((0, 4), dtype=np.float32)


        ann = dict(img_id=img_id, bbox=bboxes, cls=cls, img_size=(img_info['width'], img_info['height']))
        # ann = dict(bbox=bboxes, cls=cls, img_size=(img_info['width'],img_info['height']))


        if self.include_bboxes_ignore:
            ann['bbox_ignore'] = bboxes_ignore

        return ann

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, annotations (target)).
        """
        # img_id = self.img_ids[index]
        img_id = self.img_infos[index]['id']
        img_info = self.img_infos[index]
        if random.random() > 0.5:
            if self.has_annotations:
                ann = self._parse_img_ann(img_id, img_info)
            else:
                ann = dict(img_id=img_id, img_size=(img_info['width'], img_info['height']))
            path = img_info['filename']
            # img = Image.open(os.path.join(self.root, path)).convert('RGB')
            img = cv2.imread(os.path.join(self.root, path), cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)

            if debug:
                self.vis((np.array(img)).astype(np.uint8), ann['bbox'])
                return
            img /= 255.0
            
        else:
            img, ann = self.load_cutmix_image_and_boxes(index)
        # transform
        if self.transform is not None:
            sample = self.transform(**{'image':img, 'bboxes':ann['bbox'], 'labels':ann['cls']})
            if len(sample['bboxes']) > 0:
                img = sample['image']
                ann['bbox'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)
                ann['bbox'][:,[0,1,2,3]] = ann['bbox'][:,[1,0,3,2]]  #yxyx: be warning
        # ann = {k: v.cuda(non_blocking=True).float() for k, v in ann.items() if not isinstance(v,str)}
        return img, ann

    def __len__(self):
        return len(self.img_ids)


    def load_cutmix_image_and_boxes(self, index, imsize=1024):
        """ 
        This implementation of cutmix author:  https://www.kaggle.com/nvnnghia 
        Refactoring and adaptation: https://www.kaggle.com/shonenkov
        """
        w, h = imsize, imsize
        s = imsize // 2
    
        xc, yc = [int(random.uniform(imsize * 0.25, imsize * 0.75)) for _ in range(2)]  # center x, y
        indexes = [index] + [random.randint(0, len(self.img_ids) - 1) for _ in range(3)]

        result_image = np.full((imsize, imsize, 3), 1, dtype=np.float32)
        result_boxes = []        
        for i, index in enumerate(indexes):
            # image, boxes = self.load_image_and_boxes(index)
            img_id = self.img_infos[index]['id']
            img_info = self.img_infos[index]
            ann = self._parse_img_ann(img_id, img_info)
            path = img_info['filename']
            # image = Image.open(os.path.join(self.root, path)).convert('RGB')
            image = cv2.imread(os.path.join(self.root, path), cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
            image /= 255.0
            boxes = ann['bbox']

            if i == 0:
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
            result_image[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b]
            padw = x1a - x1b
            padh = y1a - y1b

            boxes[:, 0] += padw
            boxes[:, 1] += padh
            boxes[:, 2] += padw
            boxes[:, 3] += padh

            result_boxes.append(boxes)

        result_boxes = np.concatenate(result_boxes, 0)
        np.clip(result_boxes[:, 0:], 0, 2 * s, out=result_boxes[:, 0:])
        result_boxes = result_boxes.astype(np.float32)
        result_boxes = result_boxes[np.where((result_boxes[:,2]-result_boxes[:,0])*(result_boxes[:,3]-result_boxes[:,1]) > 0)]
        ann_mix = dict(img_id=img_id, bbox=result_boxes, cls=np.ones(result_boxes.shape[0]), img_size=(img_info['width'], img_info['height']))
        # if debug:
        #     self.vis((np.array(result_image)*255).astype(np.uint8), result_boxes)
        #     return
        # img = Image.fromarray((np.array(result_image)*255).astype(np.uint8))
        img = (np.array(result_image))
        return img, ann_mix

    def vis(self, im, boxes):
        for box in boxes:
            cv2.rectangle(im,
                        (box[0], box[1]),
                        (box[2], box[3]),
                        (220, 0, 0), 2)
        cv2.imwrite(f"./cache/test_mix.jpg", im)
