import torch
import torchvision
from torchvision import transforms
from detection.faster_rcnn import FastRCNNPredictor, FasterRCNN, fasterrcnn_resnet50_fpn
import torchvision
from torch.utils.tensorboard import SummaryWriter

import argparse
import os
import glob2
import time
import random

from utils import *
from dataset import TrainDataset, ValDataset,TestDataset,collate_fn_multi_scalse, collate_fn

def main():
    # シード値
    torch.manual_seed(7)
    torch.cuda.manual_seed(7)

    # GPU周り
    args = create_argparser().parse_args()
    device = torch.device('cuda:%d'%(args.gpu) if torch.cuda.is_available() else 'cpu')
    print(device)

    # 使用するラベル
    with open(os.path.join(args.use_dataset, "use_label.txt")) as f:
        label_name = [s.strip() for s in f.readlines()]
    use_label = ["class{}".format(i) for i in range(len(label_name)+1)]

    # 全てのラベル
    with open(os.path.join(args.use_dataset, "all_label.txt")) as f:
        all_label_name = [s.strip() for s in f.readlines()]

    # 出力ディレクトリの作成
    os.makedirs(os.path.join(args.out_foler, 'image'), exist_ok=True)
    os.makedirs(os.path.join(args.out_foler, 'model'), exist_ok=True)
    os.makedirs(os.path.join(args.out_foler, 'log'), exist_ok=True)
    summary_writer = SummaryWriter(log_dir=os.path.join(args.out_foler, 'log'))

    # # 学習条件の情報の書き込み
    write_info(args)

    ####################
    ### Make dataset ###
    ####################
    train_dataset = TrainDataset(os.path.join(args.use_dataset, "train")
                                ,args.data_root
                                ,label_name
                                ,all_label_name
                                ,args.aug
                                ,args.pixmix
                                ,args.cutmix
                                ,args.bitwise)
    val_dataset = ValDataset(os.path.join(args.use_dataset, "val")
                                ,args.data_root
                                ,label_name
                                ,all_label_name
                                ,args.bitwise)
    test_dataset = TestDataset("dataset/test/dgree3"
                                ,"JSRT/jsrt_png/Nodule154images"
                                ,label_name
                                ,all_label_name
                                ,args.bitwise)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch, shuffle=True, collate_fn=collate_fn)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    ####################
    ### Define model ###
    ####################
    num_classes = len(use_label)
    if args.backbone == "ResNet":
        model =fasterrcnn_resnet50_fpn(pretrained=False, trainable_backbone_layers=5, num_classes=num_classes)
        # model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(pretrained=True)
    else:
        model = fasterrcnn_resnet50_fpn(pretrained=False)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.to(device)

    # backbone = torchvision.models.mobilenet_v2(pretrained=True).features
    # backbone.out_channels = 1280
    # anchor_generator = AnchorGenerator(sizes=((16, 32, 64, 128, 256),),
    #                                 aspect_ratios=((0.5, 1.0, 2.0),))

    # model = FasterRCNN(backbone,
    #                 num_classes=num_classes,
    #                 rpn_anchor_generator=anchor_generator)

    # model_path = "/Work30/kitagawatomoki/part_time_job/healthcare/detection/fastrcnn_csv/result_set3/model/best_fast-rcc.pth"
    # model.load_state_dict(torch.load(model_path, map_location=device))

    params = [p for p in model.parameters() if p.requires_grad]
    if args.optimier == "SGD":
        optimizer = torch.optim.SGD(params, lr=args.learing_late)

        # lrf = 0.2
        # lf = lambda x: (1 - x / (args.epochs - 1)) * (1.0 - lrf) + lrf

        # scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
        # scheduler.last_epoch = 0 - 1  # do not move
        # optimizer = dadaptation.DadaptSGD(params, lr=args.learing_late)
        # optimizer = torch.optim.SGD(params, lr=args.learing_late, momentum=0.9, weight_decay=0.0005)
    else:
        optimizer = torch.optim.Adam(params, lr=args.learing_late)


    ##################
    ### train step ###
    ##################
    # TP, TP_FN, TP_FP = show(0, test_dataloader, model, device, os.path.join(args.out_foler, 'image'), None, None)
    print('start training')
    s_time = time.time()

    step = 0
    save_Recall = 0
    save_F1 = 0
    for epoch in range(args.epochs):
        model.train()#学習モードに移行
        s_time = time.time()

        for i, batch in enumerate(train_dataloader):

            images, targets, labels_one_hot = batch#####　batchはそのミニバッジのimage、tagets,image_idsが入ってる

            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            labels_one_hot = torch.stack(list(labels_one_hot), dim=0)
            labels_one_hot = labels_one_hot.to(device)


            ##学習モードでは画像とターゲット（ground-truth）を入力する
            ##返り値はdict[tensor]でlossが入ってる。（RPNとRCNN両方のloss）
            loss_dict= model(images, targets, labels_one_hot)

            for k, v in zip(loss_dict.keys(), loss_dict.values()):
                summary_writer.add_scalar(k, v.to('cpu').detach().numpy().copy(), step)
                # wandb.log({k: v.to('cpu').detach().numpy().copy()})

            losses = sum(loss for loss in loss_dict.values())
            summary_writer.add_scalar('train_loss', losses.to('cpu').detach().numpy().copy(), step)
            # wandb.log({'loss': losses.to('cpu').detach().numpy().copy()})
            loss_value = losses.item()

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            step+=1

            if (i+1) % 10 == 0:
                print(f"epoch #{epoch+1} Iteration #{i+1} train_loss: {loss_value}")
        # scheduler.step()

        for i, batch in enumerate(val_dataloader):

            images, targets, labels_one_hot, _ = batch#####　batchはそのミニバッジのimage、tagets,image_idsが入ってる

            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            labels_one_hot = torch.stack(list(labels_one_hot), dim=0)
            labels_one_hot = labels_one_hot.to(device)

            ##学習モードでは画像とターゲット（ground-truth）を入力する
            ##返り値はdict[tensor]でlossが入ってる。（RPNとRCNN両方のloss）
            loss_dict= model(images, targets, labels_one_hot)

            for k, v in zip(loss_dict.keys(), loss_dict.values()):
                summary_writer.add_scalar(k, v.to('cpu').detach().numpy().copy(), step)
                # wandb.log({k: v.to('cpu').detach().numpy().copy()})

            losses = sum(loss for loss in loss_dict.values())
            summary_writer.add_scalar('val_loss', losses.to('cpu').detach().numpy().copy(), step)
            # wandb.log({'loss': losses.to('cpu').detach().numpy().copy()})
            loss_value = losses.item()

            if (i+1) % 10 == 0:
                print(f"epoch #{epoch+1} Iteration #{i+1} val_loss: {loss_value}")

        print('Time for epoch {} is {}'.format(epoch+1, time.time() - s_time))

        if (epoch+1)%1 == 0:
            TP, TP_FN, TP_FP = show(epoch, test_dataloader, model, device, os.path.join(args.out_foler, 'image'), None, summary_writer)

            print('TP', TP)
            print('陽性と出た検査結果の総数', TP_FP)
            print('実際の結節の総数', TP_FN)
            if TP_FN !=0  and TP_FP != 0 and TP!=0:
                print('Precision', TP/TP_FP)
                print('Recall', TP/TP_FN)
                summary_writer.add_scalar('Precision', TP/TP_FP, step)
                summary_writer.add_scalar('Recall', TP/TP_FN, step)
                # wandb.log({'Precision': TP/TP_FP})
                # wandb.log({'Recall':TP/TP_FN})
            # else:
            #     # wandb.log({'Precision': 0})
            #     # wandb.log({'Recall':0})
                if save_Recall <= (TP/TP_FN):
                    F1 = 2*((TP/TP_FP)*(TP/TP_FN))/((TP/TP_FP)+(TP/TP_FN))
                    if save_F1 <=F1:
                        print("save_best_model : ep", epoch)
                        torch.save(model.state_dict(), os.path.join(args.out_foler, 'model', 'best_fast-rcc.pth'))
                        save_F1 = F1

                    torch.save(model.state_dict(), os.path.join(args.out_foler, 'model', 'best_recall_fast-rcc.pth'))
                    save_Recall = (TP/TP_FN)
            else:
                summary_writer.add_scalar('Precision', 0, step)
                summary_writer.add_scalar('Recall', 0, step)

        if (epoch+1)%10 == 0:
            torch.save(model.state_dict(), os.path.join(args.out_foler, 'model', 'fast-rcc_ep%d.pth'%(epoch+1)))

def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gpu", type=int, default=0)
    parser.add_argument("-data_root", type=str, default="NIH")
    parser.add_argument("--use_dataset", type=str, default="dataset/set1")
    parser.add_argument("-o", "--out_foler", type=str, default="result")
    parser.add_argument("--aug", type=int, default=1)
    parser.add_argument("--pixmix", type=int, default=0)
    parser.add_argument("--cutmix", type=int, default=1)
    parser.add_argument("--bitwise", type=int, default=0)
    parser.add_argument("--backbone", type=str, default="ResNet")
    parser.add_argument("--optimier", type=str, default="SGD")
    parser.add_argument("--learing_late", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch", type=int, default=8)

    return parser

if __name__ == '__main__':
    main()