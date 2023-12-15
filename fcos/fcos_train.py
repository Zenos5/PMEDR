"""
File: fcos_train.py
Author: Angel Wheelwright
Date: 2023-26-10
Description: Initializes or loads an FCOS detector model, trains it on a dataset, and does validation on the model.
Uses code from:
https://learnopencv.com/fcos-anchor-free-object-detection-explained/
https://github.com/rosinality/fcos-pytorch
https://github.com/VectXmy/FCOS.Pytorch/tree/master
"""

import math
import torch
from torch.utils.data import DataLoader
import os
import time
from utils.model_utils import set_train_model, get_model
from utils.checkpoint_utils import load_prog, save_prog
from utils.data_utils import loader_collate, IBEMDataset, COCODataset, getIBEMClasses
from utils.metrics_utils import validate
import argparse

def train(loader, model, optim, device):
    """
    Runs through a set of data and passes it through an FCOS detection model to train.
    Uses focal, IOU, and BCE losses for loss calculation to do backprop and optimization.

    :param loader: dataloader containing images to pass through the model
    :param model: FCOS model
    :param optim: optimizer model
    :param device: run on cpu or cuda
    :return: minimum of the cumulative loss values found during the training step
    """
    loss_val = 0.0
    model = set_train_model(model, device)
    for i, batch in enumerate(loader):
        img_list, target_list = batch
        # Training Steps
        # Clear the gradients
        optim.zero_grad()
        # Forward Pass
        # print(img_list)
        loss_dict = model(img_list, target_list)
        # print("Made it out of training!")
        loss_cls = loss_dict['classification']
        loss_box = loss_dict['bbox_regression']
        loss_center = loss_dict['bbox_ctrness']
        curr_loss = loss_cls.item() + loss_box.item() + loss_center.item()
        loss_val += curr_loss
        # Find the Loss
        loss = loss_cls + loss_box + loss_center
        # Calculate gradients
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        # Update Weights
        optim.step()

    total_loss = loss_val / (len(loader.dataset) / loader.batch_size)
    return total_loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--checkpoint", default=None, help = "path to checkpoint")
    parser.add_argument("-d", "--dataset", default="IBEM", help = "dataset to use")
    parser.add_argument("-s", "--save_rate", default=30, help = "epochs per saved checkpoint")
    parser.add_argument("-e", "--epochs", default=200, help = "Max number of epochs")
    parser.add_argument("-lr", "--learning_rate", default=1e-4, help = "Learning rate") #0.01 for SGD
    parser.add_argument("-de", "--decay", default=1e-4, help = "Learning rate")
    parser.add_argument("-b", "--batch_size", default=6, help = "Batch size")
    parser.add_argument("-n", "--run_name", default="_run", help = "Name of current run")
    parser.add_argument("-o", "--optim_name", default="adam", help = "Name of optimizer")
    parser.add_argument("-t", "--score_thresh", default=0.05, help = "Score threshold used for postprocessing the detections")
    args = parser.parse_args()
    print("Arguments:", args)

    # Create result directory.
    result_dir = 'results'
    os.makedirs(result_dir, exist_ok=True)
    # Create checkpoint directory.
    checkpoint_dir = 'checkpoints/'
    os.makedirs(checkpoint_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using {}.'.format(device))

    epochs = int(args.epochs)
    classes = None
    train_dataset = None
    val_dataset = None
    train_loader = None
    val_loader = None
    
    if args.dataset == "IBEM":
        print("Using IBEM dataset")
        classes = getIBEMClasses()

        j_file = 'IBEM/IBEM.json'
        i_dir = 'IBEM/pages/'
        
        train_lst = ['IBEM/partitions/a_cp/Tr00_cp.lst',
            'IBEM/partitions/a_cp/Tr01_cp.lst',
            'IBEM/partitions/a_cp/Tr10_cp.lst',
            'IBEM/partitions/a_cp/Ts00_cp.lst',
            'IBEM/partitions/a_cp/Ts01_cp.lst',
            'IBEM/partitions/a_cp/Ts10_cp.lst',
            'IBEM/partitions/a_cp/Va00_cp.lst'
        ] # 'IBEM/partitions/a_cp/Ts11_cp.lst' for test, 'IBEM/partitions/a_cp/Va01_cp.lst' for val
        train_dataset = IBEMDataset(lst_file=train_lst, json_file=j_file, img_dir=i_dir, device=device)
        val_dataset = IBEMDataset(lst_file=['IBEM/partitions/a_cp/Va01_cp.lst'], json_file=j_file, img_dir=i_dir, device=device)
    elif args.dataset == "COCO":
        print("Using COCO dataset")
        classes = ['__background__',
            'person',
            'bicycle',
            'car',
            'motorcycle',
            'airplane',
            'bus',
            'train',
            'truck',
            'boat',
            'traffic light',
            'fire hydrant',
            'stop sign',
            'parking meter',
            'bench',
            'bird',
            'cat',
            'dog',
            'horse',
            'sheep',
            'cow',
            'elephant',
            'bear',
            'zebra',
            'giraffe',
            'backpack',
            'umbrella',
            'handbag',
            'tie',
            'suitcase',
            'frisbee',
            'skis',
            'snowboard',
            'sports ball',
            'kite',
            'baseball bat',
            'baseball glove',
            'skateboard',
            'surfboard',
            'tennis racket',
            'bottle',
            'wine glass',
            'cup',
            'fork',
            'knife',
            'spoon',
            'bowl',
            'banana',
            'apple',
            'sandwich',
            'orange',
            'broccoli',
            'carrot',
            'hot dog',
            'pizza',
            'donut',
            'cake',
            'chair',
            'couch',
            'potted plant',
            'bed',
            'dining table',
            'toilet',
            'tv',
            'laptop',
            'mouse',
            'remote',
            'keyboard',
            'cell phone',
            'microwave',
            'oven',
            'toaster',
            'sink',
            'refrigerator',
            'book',
            'clock',
            'vase',
            'scissors',
            'teddy bear',
            'hair drier',
            'toothbrush'
        ]

        train_dataset = COCODataset("COCO", "train", device)
        val_dataset = COCODataset("COCO", "val", device)

    train_loader = DataLoader(train_dataset, batch_size=int(args.batch_size), shuffle=True, collate_fn=loader_collate)
    val_loader = DataLoader(val_dataset, batch_size=int(args.batch_size), shuffle=False, collate_fn=loader_collate)  

    print("Num classes:", len(classes))
    start_epoch = 0
    checkpoint_path = None
    if args.checkpoint is not None:
        checkpoint_path = checkpoint_dir + args.checkpoint

    checkpoint_path_out = checkpoint_dir + 'checkpoint_epoch'
    run = args.run_name #'_test'

    train_loss = []
    val_rec = []
    val_prec = []
    val_mAP = []
    val_i_prec = []
    val_i_rec = []

    model = None
    if checkpoint_path is not None:
        model, optim_weights, start_epoch, train_loss, val_prec, val_rec, val_mAP, val_i_prec, val_i_rec = load_prog(checkpoint_path, device, len(classes), run, score_thresh=float(args.score_thresh))
    else:
        model = get_model(device, num_classes=len(classes), score_thresh=float(args.score_thresh))
    print("bias:", model.state_dict()["head.classification_head.cls_logits.bias"])
    
    optim = None
    scheduler = None
    if args.optim_name == "adam":
        optim=torch.optim.Adam(model.parameters(), lr=float(args.learning_rate))
    elif args.optim_name == "sgd":
        optim=torch.optim.SGD(model.parameters(), lr=float(args.learning_rate), momentum=0.9, weight_decay=float(args.decay), nesterov=True)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[16, 22], gamma=0.1)
    else:
        optim=torch.optim.Adam(model.parameters(), lr=float(args.learning_rate))
    if checkpoint_path is not None:
        optim.load_state_dict(optim_weights)

    train_len = len(train_dataset)
    val_len = len(val_dataset)

    train_t = 0.0
    val_t = 0.0

    start_t = time.time()

    for epoch in range(start_epoch, start_epoch + epochs):
        print("EPOCH", epoch)
        print('Training')
        t_start_time = time.time()
        loss = train(train_loader, model, optim, device)
        train_loss.append(loss)
        train_t += time.time() - t_start_time
        v_start_time = time.time()
        print('Validating')
        mAP, prec, rec, i_rec, i_prec = validate(val_loader, model, device)
        val_mAP.append(mAP)
        val_rec.append(rec)
        val_prec.append(prec)
        val_i_prec.append(i_prec)
        val_i_rec.append(i_rec)
        val_t += time.time() - v_start_time

        if args.optim_name == "sgd":
            scheduler.step()

        if epoch % int(args.save_rate) == 0:
            print('Saving')
            # print("bias:", epoch, model.state_dict()["head.classification_head.cls_logits.bias"])
            c_path = checkpoint_path_out + str(epoch) + run + '.pth'
            save_prog(epoch, train_loss, val_prec, val_rec, val_mAP, val_i_prec, val_i_rec, model, optim, c_path, run)

    c_path = checkpoint_path_out + str(start_epoch + epochs - 1) + run + '.pth'
    save_prog(str(start_epoch + epochs - 1), train_loss, val_prec, val_rec, val_mAP, val_i_prec, val_i_rec, model, optim, c_path, run)

    total_t = time.time() - start_t
    train_fps = (train_len * (epochs - start_epoch)) / train_t
    val_fps = (val_len * (epochs - start_epoch)) / val_t

    print("Total train loop time:", total_t)
    print("Training FPS:", train_fps)
    print("Validation FPS:", val_fps)

    with open('results/' + run + '/results.txt', 'w') as f:
        f.write("Total train loop time: ")
        f.write(str(total_t))
        f.write("\n")
        f.write("\tTraining FPS: ")
        f.write(str(train_fps))
        f.write("\n")
        f.write("\tValidation FPS: ")
        f.write(str(val_fps))
        f.write("\n")
        f.write("\tEpochs: start at ")
        f.write(str(start_epoch))
        f.write(", total iter is ")
        f.write(str(epochs))
        f.write("\n")
        f.write("\tMin Training Loss: ")
        f.write(str(min(train_loss)))
        f.write("\n")
        f.write("\tMax Validation Precision: ")
        f.write(str(max(val_prec)))
        f.write("\n")
        f.write("\tMax Validation Recall: ")
        f.write(str(max(val_rec)))
        f.write("\n")
        f.write("\tMax Validation mAP: ")
        f.write(str(max(val_mAP)))
        f.write("\n")
