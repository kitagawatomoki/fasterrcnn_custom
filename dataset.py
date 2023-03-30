import torchvision.transforms as transforms
import torch

import pandas as pd
import os
import glob2
import xml.etree.ElementTree as ET
import albumentations as A
import numpy as np
import cv2
import random

from utils import *

class TrainDataset(torch.utils.data.Dataset):

    def __init__(self,dataset
                    ,data_root
                    ,label_name
                    ,all_label_name
                    ,is_aug
                    ,is_pixmix
                    ,is_cutmix
                    ,is_bitwise):
        super().__init__()

        self.img_size = 1280
        self.mosaic_border = [-self.img_size // 2, -self.img_size // 2]

        self.label_name = label_name
        self.all_label_name = all_label_name
        self.class_name = ["class{}".format(i) for i in range(len(label_name)+1)]

        annos = glob2.glob(os.path.join(dataset, "*.txt"))

        self.img_paths = [os.path.join(data_root, "images", "{}.jpg".format(p.split("/")[-1].split(".txt")[0])) for p in annos]

        self.annos = []
        self.shapes = []
        for i in range(len(annos)):
            # ファイルを一行ずつ読み込み
            o_labels = []
            o_shapes = []
            with open(annos[i]) as f:
                for s_line in f:
                    s_line = s_line.strip()
                    s_line = s_line.split(" ")
                    o_labels.append([self.class_name.index(s_line[0]),float(s_line[3]),float(s_line[4]),float(s_line[5]),float(s_line[6])])
                    o_shapes.append((int(s_line[1]), int(s_line[2])))
            self.annos.append(np.array(o_labels, dtype=np.float32))
            self.shapes.append(o_shapes[0])

        self.indices = range(len(self.annos))
        self.len = len(self.annos)

        self.compose_normal = A.Compose([
            A.Rotate(always_apply=False, p=0.25, limit=(-15, 15), interpolation=0),
            A.HorizontalFlip(p=0.25),
        ], bbox_params=A.BboxParams(format="pascal_voc"))

        RandomBrightnessContrast = A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, brightness_by_max=True, always_apply=False, p=0.1)
        CLAHE = A.CLAHE (clip_limit=4.0, tile_grid_size=(8, 8), always_apply=False, p=0.1)

        self.compose_contrast = A.Compose([
            CLAHE,
            RandomBrightnessContrast
        ])

        self.is_bitwise = is_bitwise
        self.is_aug = is_aug
        if is_aug:
            self.is_pixmix = is_pixmix
            if is_pixmix:
                self.mixing_set_paths = glob2.glob("/fractals_and_fvis/*/images/*.png")

            self.segmentation_root = os.path.join(data_root, "lung_segmentation_images")

            self.is_cutmix = is_cutmix
            if is_cutmix:
                self.crop_img_root = os.path.join(data_root, "cut_images", "crop_images")
                self.mask_img_root = os.path.join(data_root, "cut_images", "mask_images")

    def do_expansion(self, h, w, bboxs):
        for i in range(len(bboxs)):
            bbox = bboxs[i]
            expansion = random.randrange(10)

            if bbox[0] - expansion > 0:
                bbox[0] = bbox[0] - expansion

            if bbox[1] - expansion > 0:
                bbox[1] = bbox[1] - expansion

            if bbox[2] + expansion < w:
                bbox[2] = bbox[2] + expansion

            if bbox[3] + expansion < h:
                bbox[3] = bbox[3] + expansion

        return bboxs

    def do_pixmix(self, img):
        h, w= img.shape[0], img.shape[1]

        rnd_idx = np.random.choice(len(self.mixing_set_paths))
        mixing_pic = cv2.imread(self.mixing_set_paths[rnd_idx])
        mixing_pic = cv2.cvtColor(mixing_pic, cv2.COLOR_BGR2GRAY)
        mixing_pic = cv2.cvtColor(mixing_pic, cv2.COLOR_GRAY2RGB)
        mixing_pic = cv2.resize(mixing_pic, (w, h))

        img = (img/255).astype(np.float32)
        mixing_pic = (mixing_pic/255).astype(np.float32)
        pix_img = pixmix(img, mixing_pic)
        pix_img = np.clip(pix_img, 0, 1)*255
        pix_img = pix_img.astype(np.uint8)

        return pix_img

    def past_crop(self,label_id, crop_img_paths, img, path, ret, label, mood=0):
        add_num = random.randint(1, 2)
        past_img = img.copy()
        h, w, _ = img.shape

        if mood ==0:
            path = path.split("/")[-1].split(".jpg")[0]

            segment_img_path = os.path.join(self.segmentation_root, "mask_" +path+ ".jpg")
            if os.path.isfile(segment_img_path):
                segment_img = cv2.imread(segment_img_path)
                segment_img = cv2.cvtColor(segment_img, cv2.COLOR_BGR2GRAY)
                segment_img = cv2.resize(segment_img, (w,h))
                _,segment_img = cv2.threshold(segment_img,127,255,cv2.THRESH_BINARY)
            else:
                segment_img = np.zeros((h, w))
        else:
            segment_img = np.zeros((h, w))

        count = 0
        add_num = random.randint(1, 3)
        num = 0
        while(1):
            crop_idx = np.random.choice(len(crop_img_paths))
            crop_img = cv2.imread(crop_img_paths[crop_idx])
            crop_h, crop_w = crop_img.shape[0],crop_img.shape[1]
            crop_margin = int(crop_img_paths[crop_idx].split("/")[-1].split("-")[-2].split("m")[-1])

            if w > 0 and h >0 and crop_w > 0 and crop_h >0 and crop_w < w/2 and crop_h < h/2:
                xmin = random.randrange(crop_w, w - crop_w, 1)
                ymin = random.randrange(crop_h, h - crop_h, 1)
            else:
                break

            center_x, center_y = int(xmin+crop_w/2),int(ymin+crop_h/2)
            if len(ret) == 0:
                is_put_on = False
            else:
                iou = iou_np(np.array([xmin, ymin, xmin+crop_w, ymin+crop_h]), np.array(ret))
                if iou.all() == 0:
                    is_put_on = False
                else:
                    is_put_on = True

            if (segment_img[center_y,center_x] == 255 and not is_put_on):
                replace_img = past_img[ymin:ymin+crop_h,xmin:xmin+crop_w,:]
                mask = np.zeros_like(replace_img)
                mask = cv2.rectangle(mask, (20,20)
                                    ,(replace_img.shape[1]-20,replace_img.shape[0]-20)
                                    ,(255,255,255),-1)
                mask = cv2.GaussianBlur(mask, (51, 51), 0)
                mask = mask/255
                dst = crop_img * mask + replace_img * (1 - mask)
                crop_img = dst.astype(np.uint8)

                past_img[ymin:ymin+crop_h,xmin:xmin+crop_w,:] = crop_img
                ret += [[xmin+crop_margin, ymin+crop_margin, xmin+crop_w-crop_margin, ymin+crop_h-crop_margin]]
                label +=[label_id]
                num+=1

            if count == 10 or num == add_num:
                break

            count +=1

        return past_img, ret, label

    def past_mask(self,label_id, mask_img_paths, img, path, ret, label, mood=0):
        add_num = random.randint(1, 2)
        past_img = img.copy()
        h, w, _ = img.shape

        if mood ==0:
            path = path.split("/")[-1].split(".jpg")[0]

            segment_img_path = os.path.join(self.segmentation_root, "mask_" +path+ ".jpg")
            if os.path.isfile(segment_img_path):
                segment_img = cv2.imread(segment_img_path)
                segment_img = cv2.cvtColor(segment_img, cv2.COLOR_BGR2GRAY)
                segment_img = cv2.resize(segment_img, (w,h))
                _,segment_img = cv2.threshold(segment_img,127,255,cv2.THRESH_BINARY)
            else:
                segment_img = np.zeros((h, w))
        else:
            segment_img = np.zeros((h, w))

        count = 0
        all_count = 0
        while(1):
            try:
                mask_idx = np.random.choice(len(mask_img_paths))
                mask_img = cv2.imread(mask_img_paths[mask_idx])
                mask_h, mask_w = mask_img.shape[0],mask_img.shape[1]

                mask_xmin = int(mask_img_paths[mask_idx].split("/")[-1].split("-")[-1].split("_")[-4].split(".")[0])
                mask_ymin = int(mask_img_paths[mask_idx].split("/")[-1].split("_")[-3].split(".")[0])

                if mask_xmin + mask_w  > w or mask_ymin + mask_h > h:
                    mask_xmin = random.randrange(mask_w, int(w*0.6) - mask_w, 1)
                    mask_ymin = random.randrange(mask_h, int(h*0.6) - mask_h, 1)
                else:
                    if int(random.uniform(0,1)+0.5):
                        mask_xmin = mask_xmin + random.randint(-10,10)
                        mask_ymin = mask_ymin + random.randint(-10,10)

                iou = iou_np(np.array([mask_xmin,mask_ymin,mask_xmin+mask_w, mask_ymin+mask_h]), np.array(ret))
                if np.sum(iou) == 0 and segment_img[mask_ymin,mask_xmin] == 255:
                    mask = np.zeros_like(past_img)
                    mask[mask_ymin:mask_ymin+mask_h,mask_xmin:mask_xmin+mask_w,:] = mask_img

                    mask_img = np.where(mask < 10, 255, mask).astype(np.uint8)
                    mask = np.where(mask < 10, 255, 0).astype(np.uint8)

                    t_past_img = cv2.bitwise_and(past_img, cv2.bitwise_not(mask))
                    t_past_img = np.where(t_past_img < 10, 255, t_past_img).astype(np.uint8)

                    past_img = cv2.bitwise_and(past_img, mask)
                    margin_rate = random.randint(5, 8)/10
                    dst = cv2.addWeighted(t_past_img, margin_rate, mask_img, 1-margin_rate, 0)
                    dst = np.where(dst == 255, 0, dst).astype(np.uint8)

                    past_img = cv2.bitwise_or(past_img, dst)

                    ret += [[mask_xmin,mask_ymin,mask_xmin+mask_w, mask_ymin+mask_h]]
                    label +=[label_id]
                    count+=1
                all_count+=1

            except:
                all_count+=1
                pass
            if all_count == 10 or count == add_num:
                break

        return past_img, ret, label

    def do_cutmix(self,path,img, bbox_list, t_class):
        bbox_list = bbox_list.tolist()
        t_class = t_class.tolist()

        label = self.label_name[1]
        label_id = 2
        label = label.split(",")
        l = random.choice(label)
        mood = 1 if l in ["隠れている陰影","鎖骨に隠れる陰影"] else 0
        l = self.all_label_name.index(l)

        crop_img_paths = glob2.glob(os.path.join(self.crop_img_root, "*id{}-*.jpg".format(l)))
        mask_img_paths = glob2.glob(os.path.join(self.mask_img_root, "*id{}-*.jpg".format(l)))

        if len(crop_img_paths) !=0 and len(crop_img_paths) !=0:
            # if int(random.uniform(0,1)+0.5):
            #     img, bbox_list, t_class = self.past_crop(label_id, crop_img_paths, img, path, bbox_list, t_class, mood=mood)
            # else:
            #     img, bbox_list, t_class = self.past_mask(label_id, mask_img_paths, img, bbox_list, t_class, mood=mood)
            img, bbox_list, t_class = self.past_mask(label_id, mask_img_paths, img, path, bbox_list, t_class, mood=mood)
        elif len(crop_img_paths) ==0:
            img, bbox_list, t_class = self.past_mask(label_id, mask_img_paths, img, path, bbox_list, t_class, mood=mood)
        elif len(mask_img_paths) ==0:
            img, bbox_list, t_class = self.past_crop(label_id, crop_img_paths, img, path, bbox_list, t_class, mood=mood)

        return img ,np.array(bbox_list), np.array(t_class)

    def load_mosaic(self, index):
        # loads images in a 4-mosaic

        labels4, segments4 = [], []
        s = self.img_size
        yc, xc = [int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border]  # mosaic center x, y
        indices = [index] + random.choices(self.indices, k=3)  # 3 additional image indices
        for i, index in enumerate(indices):
            # Load image
            img_path = self.img_paths[index]
            img = cv2.imread(img_path)
            h, w = self.shapes[index] 

            # place img in img4
            if i == 0:  # top left
                img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            padw = x1a - x1b
            padh = y1a - y1b

            # Labels
            labels = self.annos[index].copy()
            if labels.size:
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padw, padh)  # normalized xywh to pixel xyxy format
            labels4.append(labels)

        # Concat/clip labels
        labels4 = np.concatenate(labels4, 0)
        for x in (labels4[:, 1:], *segments4):
            np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()

        w, h,_ = img4.shape
        img4 = cv2.resize(img4, (int(w/2), int(h/2)))
        labels4[:,1:] = labels4[:,1:]/2

        return img4, labels4

    def __getitem__(self, index):
        index = self.indices[index]

        if int(random.uniform(0,1)+0.9):
            image_anns = self.annos[index]

            while True:
                if len(image_anns)<=10:
                    break
                else:
                    index = random.choice(self.indices)
                    image_anns = self.annos[index]

            img_path = self.img_paths[index]

            img = cv2.imread(img_path)
            h, w = self.shapes[index] 


            t_class = image_anns[:,0]
            bbox_list = xywhn2xyxy(image_anns[:, 1:], w, h)
            bbox_list, t_class = check_bbox(h, w, bbox_list, t_class)

            if self.is_aug:
                if int(random.uniform(0,1)+0.5):
                    bbox_list = self.do_expansion(h, w, bbox_list)
                    bbox_list, t_class = check_bbox(h, w, bbox_list, t_class)
                    # print("aug")
                    # print(bbox_list, t_class)

                if self.is_cutmix and len(t_class) <= 3:
                    img, bbox_list, t_class = self.do_cutmix(img_path, img, bbox_list, t_class)
                    bbox_list, t_class = check_bbox(h, w, bbox_list, t_class)
                    # print("is_cutmix")
                    # print(bbox_list, t_class)

                if self.is_pixmix and int(random.uniform(0,1)+0.5):
                    img = self.do_pixmix(img)

                try:
                    aug_ret = []
                    for i in range(len(t_class)):
                        aug_ret+=[[*bbox_list[i], t_class[i]]]
                    aug_ret = np.array(aug_ret, dtype=np.uint32)

                    transformed = self.compose_normal(image=img, bboxes=aug_ret)
                    img = transformed['image']
                    aug_ret = transformed['bboxes']
                    aug_ret = np.array(aug_ret, dtype=np.uint32)

                    bbox_list = []
                    t_class = []
                    for i in range(len(aug_ret)):
                        bbox_list += [aug_ret[i][:-1]]
                        t_class +=[aug_ret[i][-1]]
                    bbox_list, t_class = check_bbox(h, w, bbox_list, t_class)
                    # print("aug_ret")
                    # print(aug_ret)
                except:
                    pass

                # transformed = self.compose_contrast(image=img)
                # img = transformed['image']


            if self.is_bitwise:
                img = cv2.bitwise_not(img)

        else:
            img, labels4 = self.load_mosaic(index)
            h, w,_  =img.shape

            t_class = labels4[:,0]
            bbox_list = labels4[:, 1:]
            bbox_list, t_class = check_bbox(h, w, bbox_list, t_class)

            if self.is_aug:
                try:
                    aug_ret = []
                    for i in range(len(t_class)):
                        aug_ret+=[[*bbox_list[i], t_class[i]]]
                    aug_ret = np.array(aug_ret, dtype=np.uint32)

                    transformed = self.compose_normal(image=img, bboxes=aug_ret)
                    img = transformed['image']
                    aug_ret = transformed['bboxes']
                    aug_ret = np.array(aug_ret, dtype=np.uint32)

                    bbox_list = []
                    t_class = []
                    for i in range(len(aug_ret)):
                        bbox_list += [aug_ret[i][:-1]]
                        t_class +=[aug_ret[i][-1]]
                    bbox_list, t_class = check_bbox(h, w, bbox_list, t_class)
                    # print("aug_ret")
                    # print(aug_ret)
                except:
                    pass

                # transformed = self.compose_contrast(image=img)
                # img = transformed['image']

            if self.is_bitwise:
                img = cv2.bitwise_not(img)

        if int(random.uniform(0,1)+0.5):
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            img = clahe.apply(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        transform = transforms.Compose([
                                        transforms.ToTensor()
        ])
        img = transform(img)

        if len(bbox_list) == 0:
            bbox_list = [0, 0, w, h]
            bbox_list = torch.tensor([bbox_list], dtype=torch.float32)
            t_class = torch.zeros((1,), dtype=torch.int64)
        else:
            bbox_list = torch.tensor(bbox_list, dtype=torch.float32)
            t_class = torch.tensor(t_class, dtype=torch.int64)

        target = {}
        target["boxes"] = bbox_list
        target["labels"]= t_class
        labels_one_hot = torch.nn.functional.one_hot(t_class, num_classes=len(self.class_name))
        # target["labels_one_hot"]= torch.clip(torch.sum(labels_one_hot, dim=0), 0, 1)
        labels_one_hot = torch.clip(torch.sum(labels_one_hot, dim=0), 0, 1).to(torch.float32)

        return img, target, labels_one_hot

    def __len__(self):
        return self.len

class ValDataset(torch.utils.data.Dataset):

    def __init__(self,dataset
                    ,data_root
                    ,label_name
                    ,all_label_name
                    ,is_bitwise
                    ,is_jpg=True):
        super().__init__()

        self.label_name = label_name
        self.all_label_name = all_label_name
        self.class_name = ["class{}".format(i) for i in range(len(label_name)+1)]

        annos = glob2.glob(os.path.join(dataset, "*.txt"))

        if is_jpg:
            self.img_paths = [os.path.join(data_root, "images", "{}.jpg".format(p.split("/")[-1].split(".txt")[0])) for p in annos]
        else:
            self.img_paths = [os.path.join(data_root, "images", "{}.png".format(p.split("/")[-1].split(".txt")[0])) for p in annos]

        self.annos = []
        self.shapes = []
        for i in range(len(annos)):
            # ファイルを一行ずつ読み込み
            o_labels = []
            o_shapes = []
            with open(annos[i]) as f:
                for s_line in f:
                    s_line = s_line.strip()
                    s_line = s_line.split(" ")
                    o_labels.append([self.class_name.index(s_line[0]),float(s_line[3]),float(s_line[4]),float(s_line[5]),float(s_line[6])])
                    o_shapes.append((int(s_line[1]), int(s_line[2])))
            self.annos.append(np.array(o_labels, dtype=np.float32))
            self.shapes.append(o_shapes[0])

        self.indices = range(len(self.annos))
        self.len = len(self.annos)

        self.is_bitwise = is_bitwise

    def __getitem__(self, index):
        index = self.indices[index]

        img_path = self.img_paths[index]

        img = cv2.imread(img_path)
        h, w = self.shapes[index] 

        image_anns = self.annos[index]

        t_class = image_anns[:,0]
        bbox_list = xywhn2xyxy(image_anns[:, 1:], w, h)
        bbox_list, t_class = check_bbox(h, w, bbox_list, t_class)

        if self.is_bitwise:
            img = cv2.bitwise_not(img)

        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        # img = clahe.apply(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        # img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        transform = transforms.Compose([
                                        transforms.ToTensor()
        ])
        img = transform(img)

        if len(bbox_list) == 0:
            bbox_list = [0, 0, w, h]
            bbox_list = torch.tensor([bbox_list], dtype=torch.float32)
            t_class = torch.zeros((1,), dtype=torch.int64)
        else:
            bbox_list = torch.tensor(bbox_list, dtype=torch.float32)
            t_class = torch.tensor(t_class, dtype=torch.int64)

        target = {}
        target["boxes"] = bbox_list
        target["labels"]= t_class
        labels_one_hot = torch.nn.functional.one_hot(t_class, num_classes=len(self.class_name))
        # target["labels_one_hot"]= torch.clip(torch.sum(labels_one_hot, dim=0), 0, 1)
        labels_one_hot = torch.clip(torch.sum(labels_one_hot, dim=0), 0, 1).to(torch.float32)


        return img, target, labels_one_hot, img_path.split('/')[-1].split(".jpg")[0]

    def __len__(self):
        return self.len


class TestDataset(torch.utils.data.Dataset):

    def __init__(self,dataset
                    ,data_root
                    ,label_name
                    ,all_label_name
                    ,is_bitwise):
        super().__init__()

        self.label_name = label_name
        self.all_label_name = all_label_name
        self.class_name = ["class{}".format(i) for i in range(len(label_name)+1)]

        annos = glob2.glob(os.path.join(dataset, "*.txt"))

        self.img_paths = [os.path.join(data_root, "{}.png".format(p.split("/")[-1].split(".txt")[0])) for p in annos]

        self.annos = []
        self.shapes = []
        for i in range(len(annos)):
            # ファイルを一行ずつ読み込み
            o_labels = []
            o_shapes = []
            with open(annos[i]) as f:
                for s_line in f:
                    s_line = s_line.strip()
                    s_line = s_line.split(" ")
                    o_labels.append([self.class_name.index(s_line[0]),float(s_line[3]),float(s_line[4]),float(s_line[5]),float(s_line[6])])
                    o_shapes.append((int(s_line[1]), int(s_line[2])))
            self.annos.append(np.array(o_labels, dtype=np.float32))
            self.shapes.append(o_shapes[0])

        self.indices = range(len(self.annos))
        self.len = len(self.annos)

        self.is_bitwise = is_bitwise

    def __getitem__(self, index):
        index = self.indices[index]

        img_path = self.img_paths[index]

        img = cv2.imread(img_path)
        h, w = self.shapes[index] 

        image_anns = self.annos[index]

        t_class = image_anns[:,0]
        bbox_list = xywhn2xyxy(image_anns[:, 1:], w, h)
        bbox_list, t_class = check_bbox(h, w, bbox_list, t_class)

        if self.is_bitwise:
            img = cv2.bitwise_not(img)

        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        # img = clahe.apply(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        # img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        transform = transforms.Compose([
                                        transforms.ToTensor()
        ])
        img = transform(img)

        if len(bbox_list) == 0:
            bbox_list = [0, 0, w, h]
            bbox_list = torch.tensor([bbox_list], dtype=torch.float32)
            t_class = torch.zeros((1,), dtype=torch.int64)
        else:
            bbox_list = torch.tensor(bbox_list, dtype=torch.float32)
            t_class = torch.tensor(t_class, dtype=torch.int64)

        target = {}
        target["boxes"] = bbox_list
        target["labels"]= t_class
        labels_one_hot = torch.nn.functional.one_hot(t_class, num_classes=len(self.class_name))
        # target["labels_one_hot"]= torch.clip(torch.sum(labels_one_hot, dim=0), 0, 1)
        labels_one_hot = torch.clip(torch.sum(labels_one_hot, dim=0), 0, 1).to(torch.float32)

        return img, target, img_path.split('/')[-1].split(".jpg")[0]

    def __len__(self):
        return self.len

def collate_fn_multi_scalse(batch):
    if int(random.uniform(0,1)+0.5):
        scale = random.choice([0.5, 1.5, 2.0])
        new_batch_img = []
        new_batch_target = []
        for i in range(len(batch)):

            img = batch[i][0].permute(1, 2, 0).numpy()
            w, h,_ = img.shape
            img = cv2.resize(img, (int(w*scale), int(h*scale)))

            transform = transforms.Compose([
                                            transforms.ToTensor()
            ])
            img = transform(img)
            new_batch_img.append(img)

            bbox_list = batch[i][1]["boxes"].numpy()

            target = {}
            target["boxes"] = torch.tensor(bbox_list*scale, dtype=torch.float32)
            target["labels"]= batch[i][1]["labels"]
            new_batch_target.append(target)

        return (new_batch_img, new_batch_target)
    else:
        return tuple(zip(*batch))

def collate_fn(batch):
    return tuple(zip(*batch))