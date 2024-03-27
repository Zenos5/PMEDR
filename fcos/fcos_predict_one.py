"""
File: fcos_predict.py
Author: Angel Wheelwright
Date: 2023-24-10
Description: Loads an FCOS detector model and evaluates it on a test dataset
"""


import torch
from torch.utils.data import DataLoader
import os
import time
from utils.model_utils import get_model, set_eval_model
from utils.checkpoint_utils import load_prog
from utils.data_utils import loader_collate, IBEMDataset, ImageDataset, getIBEMClasses
from utils.metrics_utils import validate, predict
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.utils import save_image
import numpy as np


if __name__ == "__main__":
    # Create result directory.
    result_dir = 'results'
    os.makedirs(result_dir, exist_ok=True)
    # Create checkpoint directory.
    checkpoint_dir = 'checkpoints/'
    os.makedirs(checkpoint_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using {}.'.format(device))

    classes = getIBEMClasses()

    # Path to IBEM annotations json file
    # i_file = '../data/MSE/MSE Question - Help me to solve with equivalent functions to Disjunctive.normal.form.png'
    i_file = '../data/MSE/MSE Question - Expressing Ramanujan.png'
    # Path to IBEM image directory
    b_box = [33, 158, 510, 194] # box_x_min, box_y_min, box_x_max, box_y_max
    label = 1

    test_dataset = ImageDataset(i_file, label, b_box, device=device)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=loader_collate)

    print("Dataset length:", len(test_dataset))
    print("Number of classes:", len(classes))

    checkpoint_path = checkpoint_dir + "ibem/checkpoint_epoch220_ibem.pth"

    model = None
    if checkpoint_path is not None:
        model, optim_weights, start_epoch, train_loss, val_prec, val_rec, val_mAP, val_i_prec, val_i_rec = load_prog(checkpoint_path, device, len(classes), "ibem", score_thresh=0.2)
    else:
        model = get_model(device, num_classes=len(classes), score_thresh=0.2)
    
    model = set_eval_model(model, device)
    
    test_len = len(test_dataset)

    test_t = 0.0

    start_t = time.time()

    t_start_time = time.time()
    print('Predicting')

    mAP, prec, rec, i_prec, i_rec = validate(test_loader, model, device, result_dir=result_dir)
    test_t += time.time() - t_start_time

    total_t = time.time() - start_t
    test_fps = test_len / test_t

    print("Total train loop time:", total_t)
    print("Testing FPS:", test_fps)

    with open('test_results.txt', 'w') as f:
        f.write("Total testing time: ")
        f.write(str(total_t))
        f.write("\n")
        f.write("\tTesting FPS: ")
        f.write(str(test_fps))
        f.write("\n")
