import torch
import torchvision
from torchvision import transforms
from detection.faster_rcnn import FastRCNNPredictor, FasterRCNN, fasterrcnn_resnet50_fpn

import glob2
import os
import argparse


from utils import *
from dataset import TrainDataset, TestDataset

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

parser = argparse.ArgumentParser()
parser.add_argument("-g", "--gpu", type=int, default=0)
args = parser.parse_args()

device = torch.device('cuda:%d'%(args.gpu) if torch.cuda.is_available() else 'cpu')
# device = "cpu"
print(device)

####################
####################
use_csv = "dataset/set1"
result_dir = "result"

score_threshold_list = [25, 50, 70]
####################
####################

# 使用するラベル
with open(os.path.join(use_csv, "use_label.txt")) as f:
    label_name = [s.strip() for s in f.readlines()]
use_label = ["class{}".format(i) for i in range(len(label_name)+1)]

# 全てのラベル
with open(os.path.join(use_csv, "all_label.txt")) as f:
    all_label_name = [s.strip() for s in f.readlines()]

num_classes = len(use_label)

model = fasterrcnn_resnet50_fpn(pretrained=True)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
model.to(device)

model_path = "{}/model/best_recall_fast-rcc.pth".format(result_dir)
is_file = os.path.isfile(model_path)

for score_threshold in score_threshold_list:
    for dgree in reversed(range(5)):
        dgree = dgree+1
        os.makedirs('%s/eval_jsrt/eval_%d/%s/%s'%(result_dir,score_threshold, model_path.split("/")[1], dgree), exist_ok=True)

        val_dataset = TestDataset("dataset/test/dgree{}".format(dgree)
                                    ,"/JSRT/jsrt_png/Nodule154images"
                                    ,label_name
                                    ,all_label_name
                                    ,False)

        def collate_fn(batch):
            return tuple(zip(*batch))

        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

        if is_file:
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()


            label_dict = dict()
            predict_dict = dict()


            for a , batch in enumerate(val_dataloader):

                images, targets, f_name = batch

                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                imgs = images
                img = imgs[0].permute(1, 2, 0).cpu().numpy()
                labelsets = targets[0]


                label_dict[f_name[0]] = []
                label_img = img.copy()
                label_img = (label_img*255).astype(np.uint8)
                label_img = cv2.putText(label_img, 'Label:JSRT dgree%s'%(dgree),
                                    (0,120),
                                    cv2.FONT_HERSHEY_PLAIN,
                                    10,
                                    (255, 0, 0),
                                    5,
                                    cv2.LINE_AA)

                for i in range(labelsets['boxes'].size()[0]):
                    xmin = int(labelsets["boxes"][i][0].cpu().numpy())
                    ymin = int(labelsets["boxes"][i][1].cpu().numpy())
                    xmax = int(labelsets["boxes"][i][2].cpu().numpy())
                    ymax = int(labelsets["boxes"][i][3].cpu().numpy())
                    label_dict[f_name[0]].append({"label":int(labelsets["labels"].cpu().numpy()),
                                                "xmin":xmin,
                                                "ymin":ymin,
                                                "xmax":xmax,
                                                "ymax":ymax})

                    label_img = cv2.putText(label_img, 'Nodule',
                                        (xmin,ymin-20), 
                                        cv2.FONT_HERSHEY_PLAIN, 
                                        5, 
                                        (255, 0, 0), 
                                        5, 
                                        cv2.LINE_AA)

                    label_img = cv2.rectangle(label_img, (xmin,ymin)
                                        ,(xmax, ymax)
                                        ,(255,0,0),3)
                    label_img = label_img.clip(0, 255)
                # plt.figure(figsize=(15,12))
                # plt.imshow(label_img)
                # plt.show()

                outputs = model(images)
                outputs = outputs[0]

                output_box = outputs['boxes'].detach().cpu().numpy()
                output_box = output_box.astype(np.uint64)

                output_labels = outputs['labels'].detach().cpu().numpy()
                output_scores = outputs['scores'].detach().cpu().numpy()

                output_box = output_box.tolist()
                output_labels = output_labels.tolist()
                output_scores = output_scores.tolist()

                # output_box, output_scores, output_labels = soft_nms(output_box, output_scores, output_labels, iou_threshold=0.1, sigma=0.05, linear=False)
                output_box, output_scores, output_labels = soft_nms(output_box, output_scores, output_labels, iou_threshold=0.1, sigma=0.05, linear=True)

                new_box = []
                new_labels = []
                new_scores = []
                for i in range(len(output_box)):
                    if output_scores[i] >=0.5:
                        new_box.append(output_box[i])
                        new_labels.append(output_labels[i])
                        new_scores.append(output_scores[i])

                output_box = new_box
                output_labels = new_labels
                output_scores = new_scores

                remove_index = []
                for a_n, a_bbox in enumerate(output_box):
                    for b_n, b_bbox in enumerate(output_box):
                        iou, position = one_iou_np(a_bbox, b_bbox)
                        if position and a_n!=b_n and abs(iou)!=1:
                            # print(position, iou, a_n, b_n)
                            remove_index.append(b_n)
                            # print(b_bbox)
                            # print(output_scores[b_n])

                remove_index = list(dict.fromkeys(remove_index))

                output_box = np.delete(output_box, remove_index, 0)
                output_labels = np.delete(output_labels, remove_index, 0)
                output_scores = np.delete(output_scores, remove_index, 0)


                num_box = len(output_box)
                predict_dict[f_name[0]] = []
                predict_img = img.copy()
                predict_img = (predict_img*255).astype(np.uint8)
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
                    # if n_score>=50:
                    if n_score>=score_threshold:
                        predict_dict[f_name[0]].append({"label":n_label,
                                                        "score":n_score,
                                                    "xmin":xmin,
                                                    "ymin":ymin,
                                                    "xmax":xmax,
                                                    "ymax":ymax})

                        predict_img = cv2.putText(predict_img, '{} : {}%'.format(use_label[n_label], n_score),
                                            (xmin,ymin-20),
                                            cv2.FONT_HERSHEY_PLAIN,
                                            2,
                                            (0, 0, 255),
                                            2,
                                            cv2.LINE_AA)

                        predict_img = cv2.rectangle(predict_img, (xmin,ymin)
                                            ,(xmax, ymax)
                                            ,(0,0,255),3)
                        predict_img = predict_img.clip(0, 255)

                show_img = np.concatenate([label_img, predict_img], 1)

                cv2.imwrite("%s/eval_jsrt/eval_%d/%s/%s/%s.png"%(result_dir,score_threshold,model_path.split("/")[1], dgree,f_name[0]), show_img)

            write_list = []
            for i in range(num_classes-1):
                TP_list = []
                TP_FN_list = []
                TP_FP_list = []
                for k, v in label_dict.items():
                    TP_FN = len(v)

                    predict_boxes=[]
                    TP_FP = 0
                    for info in predict_dict[k]:
                        # if info["label"] ==1 and info["score"]>=50:
                        if info["score"]>=score_threshold and info["label"] ==i+1:
                            predict_boxes.append([info["xmin"], info["ymin"], info["xmax"], info["ymax"]])
                            TP_FP+=1
                        # predict_boxes.append([info["xmin"], info["ymin"], info["xmax"], info["ymax"]])
                    predict_boxes = np.array(predict_boxes)

                    # TP_FP = len(predict_boxes)
                    TP = 0
                    if TP_FP != 0:
                        for info in v:
                            label_box = np.array([info["xmin"], info["ymin"], info["xmax"], info["ymax"]])

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

                # print('TP', TP)
                # print('陽性と出た検査結果の総数', TP_FP)
                # print('実際の結節の総数', TP_FN)
                if TP_FN !=0  and TP_FP != 0:
                    Precision = TP/TP_FP
                    Recall = TP/TP_FN
                else:
                    Precision = 0
                    Recall = 0

                write_text = "class:{} | dgree:{} | Recall:{:.2f} | Precision:{:.2f} | TP:{} | TP_FP:{} | TP_FN:{}".format(use_label[i+1],dgree, Recall, Precision, TP, TP_FP, TP_FN)
                write_list.append(write_text)


            TP_list = []
            TP_FN_list = []
            TP_FP_list = []
            for k, v in label_dict.items():
                TP_FN = len(v)

                predict_boxes=[]
                TP_FP = 0
                for info in predict_dict[k]:
                    # if info["label"] ==1 and info["score"]>=50:
                    if info["score"]>=score_threshold:
                        predict_boxes.append([info["xmin"], info["ymin"], info["xmax"], info["ymax"]])
                        TP_FP+=1
                    # predict_boxes.append([info["xmin"], info["ymin"], info["xmax"], info["ymax"]])
                predict_boxes = np.array(predict_boxes)

                # TP_FP = len(predict_boxes)
                TP = 0
                if TP_FP != 0:
                    for info in v:
                        label_box = np.array([info["xmin"], info["ymin"], info["xmax"], info["ymax"]])

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

            # print('TP', TP)
            # print('陽性と出た検査結果の総数', TP_FP)
            # print('実際の結節の総数', TP_FN)
            if TP_FN !=0  and TP_FP != 0:
                Precision = TP/TP_FP
                Recall = TP/TP_FN
            else:
                Precision = 0
                Recall = 0

            write_text = "class:all | dgree:{} | Recall:{:.2f} | Precision:{:.2f} | TP:{} | TP_FP:{} | TP_FN:{}".format(dgree, Recall, Precision, TP, TP_FP, TP_FN)
            write_list.append(write_text)

            write_path = "%s/eval_jsrt/eval_%d/result_%d.txt"%(result_dir,score_threshold, dgree)
            # リストを書き込み
            with open(write_path, mode='w') as f:
                f.write('\n'.join(write_list))
