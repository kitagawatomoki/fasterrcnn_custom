
import os


def write_info(args):
    write_list = []

    write_list.append("data_root:{}".format(args.data_root))
    write_list.append("use_dataset:{}".format(args.use_dataset))
    write_list.append("aug:{}".format(args.aug))
    write_list.append("pixmix:{}".format(args.pixmix))
    write_list.append("cutmix:{}".format(args.cutmix))
    write_list.append("bitwise:{}".format(args.bitwise))
    write_list.append("backbone:{}".format(args.backbone))
    write_list.append("optimier:{}".format(args.optimier))
    write_list.append("learing_late:{}".format(args.learing_late))
    write_list.append("epochs:{}".format(args.epochs))
    write_list.append("batch:{}".format(args.batch))

    with open(os.path.join(args.out_foler, "information.txt"), mode='w') as f:
        f.write('\n'.join(write_list))


from pixmix_utils import *

def pixmix(orig, mixing_pic, k=3, beta=5):
    # mixings = utils.mixings
    mixings = [add, multiply]
    mixed = orig

    for _ in range(np.random.randint(k + 1)):
        if np.random.random() < 0.5:
            aug_image_copy = orig
        else:
            aug_image_copy = mixing_pic

        mixed_op = np.random.choice(mixings)
        mixed = mixed_op(mixed, aug_image_copy, beta)

    return mixed

def check_bbox(h, w, bboxs, labels):
    select_bboxs = []
    select_labels = []
    for i in range(len(labels)):
        bbox = bboxs[i]
        label = labels[i]
        if bbox[0] != bbox[2] and bbox[1] != bbox[3]:
            if bbox[0]<0:
                bbox[0] = 0

            if bbox[1]<0:
                bbox[1] = 0

            if bbox[2]>w:
                bbox[2] = w

            if bbox[3]>h:
                bbox[3] = h

            select_bboxs += [bbox]
            select_labels += [label]

    return np.array(select_bboxs, dtype=np.float32), np.array(select_labels, dtype=np.int64)

import numpy as np

# 矩形aと、複数の矩形bのIoUを計算
def iou_np(a, b):
    # aは1つの矩形を表すshape=(4,)のnumpy配列
    # array([xmin, ymin, xmax, ymax])
    # bは任意のN個の矩形を表すshape=(N, 4)のnumpy配列
    # 2次元目の4は、array([xmin, ymin, xmax, ymax])

    # 矩形aの面積a_areaを計算
    a_area = (a[2] - a[0] + 1) \
             * (a[3] - a[1] + 1)
    # bに含まれる矩形のそれぞれの面積b_areaを計算
    # shape=(N,)のnumpy配列。Nは矩形の数
    b_area = (b[:,2] - b[:,0] + 1) \
             * (b[:,3] - b[:,1] + 1)
    # aとbの矩形の共通部分(intersection)の面積を計算するために、
    # N個のbについて、aとの共通部分のxmin, ymin, xmax, ymaxを一気に計算
    abx_mn = np.maximum(a[0], b[:,0]) # xmin
    aby_mn = np.maximum(a[1], b[:,1]) # ymin
    abx_mx = np.minimum(a[2], b[:,2]) # xmax
    aby_mx = np.minimum(a[3], b[:,3]) # ymax
    # 共通部分の矩形の幅を計算。共通部分が無ければ0
    w = np.maximum(0, abx_mx - abx_mn + 1)
    # 共通部分の矩形の高さを計算。共通部分が無ければ0
    h = np.maximum(0, aby_mx - aby_mn + 1)
    # 共通部分の面積を計算。共通部分が無ければ0
    intersect = w*h

    # N個のbについて、aとのIoUを一気に計算
    iou = intersect / (a_area + b_area - intersect)
    return iou

# 矩形aと、複数の矩形bのIoUを計算
def one_iou_np(a, b):
    # 矩形aの面積a_areaを計算
    a_area = (a[2] - a[0] + 1) \
             * (a[3] - a[1] + 1)
    # bに含まれる矩形のそれぞれの面積b_areaを計算
    b_area = (b[2] - b[0] + 1) \
             * (b[3] - b[1] + 1)
    # aとbの矩形の共通部分(intersection)の面積を計算するために、
    # N個のbについて、aとの共通部分のxmin, ymin, xmax, ymaxを一気に計算
    abx_mn = np.maximum(a[0], b[0]) # xmin
    aby_mn = np.maximum(a[1], b[1]) # ymin
    abx_mx = np.minimum(a[2], b[2]) # xmax
    aby_mx = np.minimum(a[3], b[3]) # ymax
    # 共通部分の矩形の幅を計算。共通部分が無ければ0
    w = np.maximum(0, abx_mx - abx_mn + 1)
    # 共通部分の矩形の高さを計算。共通部分が無ければ0
    h = np.maximum(0, aby_mx - aby_mn + 1)
    # 共通部分の面積を計算。共通部分が無ければ0
    intersect = w*h

    # N個のbについて、aとのIoUを一気に計算
    iou = intersect / (a_area + b_area - intersect)

    if intersect == b_area:
        position = True
    else:
        position = False

    return iou, position

def iou(a, b):
    # a, bは矩形を表すリストで、a=[xmin, ymin, xmax, ymax]
    ax_mn, ay_mn, ax_mx, ay_mx = a[0], a[1], a[2], a[3]
    bx_mn, by_mn, bx_mx, by_mx = b[0], b[1], b[2], b[3]

    a_area = (ax_mx - ax_mn + 1) * (ay_mx - ay_mn + 1)
    b_area = (bx_mx - bx_mn + 1) * (by_mx - by_mn + 1)

    abx_mn = max(ax_mn, bx_mn)
    aby_mn = max(ay_mn, by_mn)
    abx_mx = min(ax_mx, bx_mx)
    aby_mx = min(ay_mx, by_mx)
    w = max(0, abx_mx - abx_mn + 1)
    h = max(0, aby_mx - aby_mn + 1)
    intersect = w*h

    iou = intersect / (a_area + b_area - intersect)
    return iou


# スコアの更新関数（線形）
def f_linear(iou, iou_threshold=0.5):
    if iou >= iou_threshold:
        weight = 1. - iou
    else:
        weight = 1.
    return weight

# スコアの更新関数（指数関数、gauss関数）
def f_gauss(iou, sigma=0.5):
    import math
    return math.exp(-iou*iou/sigma)

def soft_nms(bboxes, scores, classes, \
             iou_threshold=0.5, sigma=0.5, linear=True):
    new_bboxes = [] # Soft-NMS適用後の矩形リスト
    new_scores = [] # Soft-NMS適用後の信頼度(スコア値)リスト
    new_classes = [] # Soft-NMS適用後のクラスのリスト

    while len(bboxes) > 0:
        # スコア最大の矩形のインデックスを取得
        argmax = scores.index(max(scores))

        # スコア最大の矩形、スコア値、クラスをそれぞれのリストから消去
        bbox = bboxes.pop(argmax)
        score = scores.pop(argmax)
        clss = classes.pop(argmax)

        # スコア最大の矩形と、対応するスコア値、クラスをSoft-NMS適用後のリストに格納
        new_bboxes.append(bbox)
        new_scores.append(score)
        new_classes.append(clss)

        # bboxesに残存する矩形のスコアを更新
        for i, bbox_tmp in enumerate(bboxes):
            # スコア最大の矩形bboxと他の矩形のIoUを計算
            iou_tmp = iou(bbox, bbox_tmp)
            # scoreの値を更新
            if linear: 
                # スコアの更新関数（線形）
                scores[i] = scores[i]*f_linear(iou_tmp, iou_threshold)
            else: 
                # スコアの更新関数（指数関数、gauss関数）
                scores[i] = scores[i]*f_gauss(iou_tmp, sigma)

    return new_bboxes, new_scores, new_classes


def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + padw  # top left x
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + padh  # top left y
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + padw  # bottom right x
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + padh  # bottom right y
    return y


import cv2
import wandb

def cvt2HeatmapImg(img):
    img = img.clip(0,255)
    img = img.astype(np.uint8)
    img = 255-cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = cv2.applyColorMap(img, cv2.COLORMAP_JET)/255
    img = img.astype(np.float32)
    return img


def show(epoch, val_dataloader, model, device, save_dir, run=None, writer=None):
    model.to(device)
    model.eval()#推論モードへ

    label_dict = dict()
    predict_dict = dict()

    for a , batch in enumerate(val_dataloader):

        images, targets, f_name = batch

        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        imgs = images
        img = imgs[0].permute(1, 2, 0).cpu().numpy()
        labelsets = targets
        # print(len(labelsets), labelsets[0]['labels'])
        label_img = img.copy()

        label_bbox = []
        for i in range(labelsets[0]['boxes'].size()[0]):
            if int(labelsets[0]['labels'][i].cpu().numpy())== 1:
                ret = labelsets[0]['boxes'][i].cpu().numpy()
                ret = [int(n) for n in ret]
                label_bbox.append(ret)

                label_img = cv2.rectangle(label_img, (ret[0],ret[1])
                                    ,(ret[2], ret[3])
                                    ,(255,0,0),3)
                label_img = label_img.clip(0, 1)

        if len(label_bbox) !=0:
            label_dict[f_name[0]] = label_bbox

        outputs = model(images)
        outputs = outputs[0]

        predict_img = (img.copy()).astype(np.float32)
        heat_w, heat_h, _ = predict_img.shape

        # num_box = len(outputs[0]['boxes'])
        output_box = outputs['boxes'].detach().cpu().numpy()
        output_box = output_box.astype(np.uint64)

        output_labels = outputs['labels'].detach().cpu().numpy()
        output_scores = outputs['scores'].detach().cpu().numpy()

        output_box = output_box.tolist()
        output_labels = output_labels.tolist()
        output_scores = output_scores.tolist()

        output_box, output_scores, output_labels = soft_nms(output_box, output_scores, output_labels, iou_threshold=0.1, sigma=0.5, linear=True)
        num_box = len(output_box)

        if len(output_box)==0:
            predict_dict[f_name[0]] = []
            box_img = cv2.putText(predict_img.copy(),
                                'No detection',
                                (int(heat_h/4),int(heat_h/2)), 
                                cv2.FONT_HERSHEY_PLAIN, 
                                5, 
                                (255, 0, 0), 
                                5, 
                                cv2.LINE_AA)

        else:
            num_box = len(output_box)
            predict_dict[f_name[0]] = []
            # predict_img = (predict_img*255).astype(np.uint8)
            predict_img = cv2.putText(predict_img, 'AI',
                                (0,120),
                                cv2.FONT_HERSHEY_PLAIN,
                                10,
                                (0, 0, 255),
                                5,
                                cv2.LINE_AA)
            for n in range(num_box):
                n_label = output_labels[n]
                n_score = int(output_scores[n]*100)
                xmin = int(output_box[n][0])
                ymin = int(output_box[n][1])
                xmax = int(output_box[n][2])
                ymax = int(output_box[n][3])

                predict_dict[f_name[0]].append({"label":n_label,
                                                "score":n_score,
                                            "xmin":xmin,
                                            "ymin":ymin,
                                            "xmax":xmax,
                                            "ymax":ymax})

                predict_img = cv2.putText(predict_img, '{} : {}%'.format(n_label, n_score),
                                    (xmin,ymin-20),
                                    cv2.FONT_HERSHEY_PLAIN,
                                    2,
                                    (0, 0, 255),
                                    2,
                                    cv2.LINE_AA)

                predict_img = cv2.rectangle(predict_img, (xmin,ymin)
                                    ,(xmax, ymax)
                                    ,(0,0,255),3)
            box_img = predict_img.clip(0, 1)

        # display_img = (np.concatenate([label_img, box_img, blended], 1)*255).astype('uint8')
        display_img = (np.concatenate([label_img, box_img], 1)*255).astype('uint8')

        cv2.imwrite(os.path.join(save_dir, 'resample_test%d.png'%(a)), cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB))
        if run != None:
            images = wandb.Image(display_img, caption="Top: Output, Bottom: Input")
            run.log({"test %s"%(a): images})

        image = np.transpose(display_img/255, (2,0,1))
        #画像を保存したい時用
        if writer != None:
            writer.add_image('resample_test%d'%(a), image, epoch)

    TP_list = []
    TP_FN_list = []
    TP_FP_list = []
    for k, v in label_dict.items():
        TP_FN = len(v)

        predict_boxes=[]
        TP_FP = 0
        for info in predict_dict[k]:
            # if info["label"] ==1 and info["score"]>=50:
            if info["score"]>=50:
                predict_boxes.append([info["xmin"], info["ymin"], info["xmax"], info["ymax"]])
                TP_FP+=1
            # predict_boxes.append([info["xmin"], info["ymin"], info["xmax"], info["ymax"]])
        predict_boxes = np.array(predict_boxes)

        # TP_FP = len(predict_boxes)
        TP = 0
        if TP_FP != 0:
            for info in v:
                label_box = np.array(info)

                iou = iou_np(label_box, predict_boxes)
                if iou.max() >=0.1:
                    # print(f_name)
                    # print(iou.max())
                    TP+=1

            TP_list.append(TP)
            TP_FP_list.append(TP_FP)
        TP_FN_list.append(TP_FN)

    TP = sum(TP_list)
    TP_FN = sum(TP_FN_list)
    TP_FP = sum(TP_FP_list)

    return TP, TP_FN, TP_FP
