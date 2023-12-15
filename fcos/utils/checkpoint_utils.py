"""
File: checkpoint_utils.py
Author: Angel Wheelwright
Date: 2023-24-10
Description: Contains utility functions for making and loading checkpoints
"""

import torch
import json
from utils.model_utils import get_model

def save_prog(epoch, train_loss, val_prec, val_rec, val_mAP, val_i_prec, val_i_rec, model, optim, path, run):
    """
    Creates a checkpoint file containing the current epoch, the training losses and validation precision, 
    recall, and mean average precision (mAP) for each epoch iterated through previously, and a copy of the model 
    and optimizer at that epoch.

    :param epoch: current epoch reached during training
    :param train_loss: list of the last loss value calculated for each epoch
        The loss is composed of the sum of the classification, bounding box regression, and bounding box centeredness losses, 
        which are focal loss, IoU loss, and binary cross-entropy error (BCE) loss respectively
    :param val_prec: list of the average precision calculated for each epoch
    :param val_rec: list of the average recall calculated for each epoch
    :param val_mAP: list of the average mAP calculated for each epoch
    :param model: FCOS model
    :param optim: optimizer model
    :param path: path to the file location with name used to save the checkpoint file to
    """ 
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optim.state_dict()
            }, path)
    name = 'checkpoints/'+ run + '/'
    with open(name + 'train_loss.txt', 'w') as fp:
        fp.write("\n".join(str(item) for item in train_loss))
    with open(name + 'val_prec.txt', 'w') as fp:
        fp.write("\n".join(str(item) for item in val_prec))
    with open(name + 'val_rec.txt', 'w') as fp:
        fp.write("\n".join(str(item) for item in val_rec))
    with open(name + 'val_mAP.txt', 'w') as fp:
        fp.write("\n".join(str(item) for item in val_mAP))
    with open(name + 'val_i_prec.txt', 'w') as fp:
        json.dump(val_i_prec, fp)
    with open(name + 'val_i_rec.txt', 'w') as fp:
        json.dump(val_i_rec, fp)

def load_prog(path, device, num_classes, run, score_thresh):
    """
    Loads a checkpoint file containing the current epoch, the training losses and validation precision, 
    recall, and mean average precision (mAP) for each epoch iterated through previously, and a copy of the model 
    and optimizer at that epoch.

    :param path: path to the file location with name used to load the checkpoint file from
    :param model: FCOS model to update
    :param optim: optimizer model to update
    :return: list of images, list of annotations
        model: FCOS model
        optim: optimizer model
        start_epoch: epoch to start training at
        train_loss: list of the last loss value calculated for each epoch
            The loss is composed of the sum of the classification, bounding box regression, and bounding box centeredness losses, 
            which are focal loss, IoU loss, and binary cross-entropy error (BCE) loss respectively
        val_prec: list of the average precision calculated for each epoch
        val_rec: list of the average recall calculated for each epoch
        val_mAP: list of the average mAP calculated for each epoch
    """ 
    checkpoint = torch.load(path)
    model = get_model(device, num_classes=num_classes, score_thresh=score_thresh)
    print("Current:", model.state_dict()['head.classification_head.cls_logits.bias'])
    print("Replace with:", checkpoint['model_state_dict']['head.classification_head.cls_logits.bias'])
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    start_epoch = int(epoch) + 1
    train_loss = []
    val_prec = []
    val_rec = []
    val_mAP = []
    val_i_prec = []
    val_i_rec = []
    name = 'checkpoints/'+ run + '/'
    with open(name + 'train_loss.txt', 'r') as fp:
        for line in fp:
            train_loss.append(float(line.strip()))
    with open(name + 'val_prec.txt', 'r') as fp:
        for line in fp:
            val_prec.append(float(line.strip()))
    with open(name + 'val_rec.txt', 'r') as fp:
        for line in fp:
            val_rec.append(float(line.strip()))
    with open(name + 'val_mAP.txt', 'r') as fp:
        for line in fp:
            val_mAP.append(float(line.strip()))
    with open(name + 'val_i_prec.txt', 'r') as fp:
        val_i_prec = json.load(fp)
    with open(name + 'val_i_rec.txt', 'r') as fp:
        val_i_rec = json.load(fp)
    return model, checkpoint['optimizer_state_dict'], start_epoch, train_loss, val_prec, val_rec, val_mAP, val_i_prec, val_i_rec