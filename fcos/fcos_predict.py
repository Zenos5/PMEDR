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
from utils.data_utils import loader_collate, IBEMDataset, COCODataset, getIBEMClasses
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

    BATCH_SIZE=6

    # Path to IBEM annotations json file
    j_file = 'IBEM/IBEM.json'
    # Path to IBEM image directory
    i_dir = 'IBEM/pages/'

    test_dataset = IBEMDataset(lst_file=['IBEM/partitions/a_cp/Ts11_cp.lst'], json_file=j_file, img_dir=i_dir, device=device)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=loader_collate)

    # classes = [
    #     '__background__',
    #     'person',
    #     'bicycle',
    #     'car',
    #     'motorcycle',
    #     'airplane',
    #     'bus',
    #     'train',
    #     'truck',
    #     'boat',
    #     'traffic light',
    #     'fire hydrant',
    #     'stop sign',
    #     'parking meter',
    #     'bench',
    #     'bird',
    #     'cat',
    #     'dog',
    #     'horse',
    #     'sheep',
    #     'cow',
    #     'elephant',
    #     'bear',
    #     'zebra',
    #     'giraffe',
    #     'backpack',
    #     'umbrella',
    #     'handbag',
    #     'tie',
    #     'suitcase',
    #     'frisbee',
    #     'skis',
    #     'snowboard',
    #     'sports ball',
    #     'kite',
    #     'baseball bat',
    #     'baseball glove',
    #     'skateboard',
    #     'surfboard',
    #     'tennis racket',
    #     'bottle',
    #     'wine glass',
    #     'cup',
    #     'fork',
    #     'knife',
    #     'spoon',
    #     'bowl',
    #     'banana',
    #     'apple',
    #     'sandwich',
    #     'orange',
    #     'broccoli',
    #     'carrot',
    #     'hot dog',
    #     'pizza',
    #     'donut',
    #     'cake',
    #     'chair',
    #     'couch',
    #     'potted plant',
    #     'bed',
    #     'dining table',
    #     'toilet',
    #     'tv',
    #     'laptop',
    #     'mouse',
    #     'remote',
    #     'keyboard',
    #     'cell phone',
    #     'microwave',
    #     'oven',
    #     'toaster',
    #     'sink',
    #     'refrigerator',
    #     'book',
    #     'clock',
    #     'vase',
    #     'scissors',
    #     'teddy bear',
    #     'hair drier',
    #     'toothbrush'
    # ]
    # test_dataset = COCODataset("COCO", "val", device)

    # test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=loader_collate)

    print("Dataset length:", len(test_dataset))
    print("Number of classes:", len(classes))

    checkpoint_path = checkpoint_dir + "_ibem/checkpoint_epoch100_ibem.pth"

    model = None
    if checkpoint_path is not None:
        model, optim_weights, start_epoch, train_loss, val_prec, val_rec, val_mAP, val_i_prec, val_i_rec = load_prog(checkpoint_path, device, len(classes), "_ibem", score_thresh=0.2)
    else:
        model = get_model(device, num_classes=len(classes), score_thresh=0.2)
    
    model = set_eval_model(model, device)
    
    test_len = len(test_dataset)

    test_t = 0.0

    start_t = time.time()

    t_start_time = time.time()
    print('Predicting')
    # for i, batch in enumerate(test_loader):
    #     # timer = time.time()
    #     img_list, target_list = batch

    #     boxes, labels, scores = predict(img_list, model, 0.5)
    #     print(i, "=", boxes, labels, scores)
    
    # test_dataset = IBEMDataset(lst_file=['IBEM/partitions/a_cp/Ts00_cp.lst','IBEM/partitions/a_cp/Ts01_cp.lst'], json_file=j_file, img_dir=i_dir, device=device)
    
    # test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=loader_collate)

    # test_dataset = COCODataset("COCO", "val", device)

    # test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=loader_collate)

    mAP, prec, rec, i_prec, i_rec = validate(test_loader, model, device, result_dir=result_dir)
    test_t += time.time() - t_start_time

    total_t = time.time() - start_t
    test_fps = test_len / test_t

    print("Total train loop time:", total_t)
    print("Testing FPS:", test_fps)

    prec_1 = 0.0
    prec_1_count = 0
    prec_5 = 0.0
    prec_5_count = 0
    prec_10 = 0.0
    prec_10_count = 0
    for item in i_prec:
        if len(item) > 1:
            prec_1 += item[1]
            prec_1_count += 1
        if len(item) > 6:
            prec_5 += np.mean(item[1:6])
            prec_5_count += 1
        if len(item) > 11:
            prec_10 += np.mean(item[1:11])
            prec_10_count += 1
    prec_1 /= prec_1_count
    prec_5 /= prec_5_count
    prec_10 /= prec_10_count

    rec_1 = 0.0
    rec_1_count = 0
    rec_5 = 0.0
    rec_5_count = 0
    rec_10 = 0.0
    rec_10_count = 0
    for item in i_rec:
        if len(item) > 1:
            rec_1 += item[1]
            rec_1_count += 1
        if len(item) > 6:
            rec_5 += np.mean(item[1:6])
            rec_5_count += 1
        if len(item) > 11:
            rec_10 += np.mean(item[1:11])
            rec_10_count += 1
    rec_1 /= rec_1_count
    rec_5 /= rec_5_count
    rec_10 /= rec_10_count

    with open('test_results.txt', 'w') as f:
        f.write("Total testing time: ")
        f.write(str(total_t))
        f.write("\n")
        f.write("\tTesting FPS: ")
        f.write(str(test_fps))
        f.write("\n")
        f.write("\tPrediction Precision: ")
        f.write(str(prec))
        f.write("\n")
        f.write("\tPrediction Recall: ")
        f.write(str(rec))
        f.write("\n")
        f.write("\tPrediction mAP: ")
        f.write(str(mAP))
        f.write("\n")
        f.write("\tPrediction Precision@1: ")
        f.write(str(prec_1))
        f.write("\n")
        f.write("\tPrediction Recall@1: ")
        f.write(str(rec_1))
        f.write("\n")
        f.write("\tPrediction Precision@5: ")
        f.write(str(prec_5))
        f.write("\n")
        f.write("\tPrediction Recall@5: ")
        f.write(str(rec_5))
        f.write("\n")
        f.write("\tPrediction Precision@10: ")
        f.write(str(prec_10))
        f.write("\n")
        f.write("\tPrediction Recall@10: ")
        f.write(str(rec_10))
        f.write("\n")
