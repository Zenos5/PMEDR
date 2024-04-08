"""
File: metrics_utils.py
Author: Angel Wheelwright
Date: 2023-24-10
Description: Contains utility functions for doing predictions and computing error metrics for the model
Uses code from:
https://github.com/rafaelpadilla/Object-Detection-Metrics#metrics
https://learnopencv.com/fcos-anchor-free-object-detection-explained/
"""

import torch
import math
import numpy as np
import cv2
import os
import sys
from utils.data_utils import getIBEMClasses
from utils.model_utils import set_eval_model
from utils.display_utils import draw_boxes

@staticmethod
def calc_iou(boxA, boxB):
    """
    Calculates Intersection over Union (IoU) loss for a pair of bounding boxes.
    IoU measures the overlap between predicted bounding boxes and ground truth boxes, with scores ranging from 0 to 1.

    :param boxA: groundtruth bounding box
    :param boxB: predicted bounding box
    :return: 
        iou: IoU loss between the bounding boxes
    """ 
    # if boxes dont intersect
    if boxes_intersect(boxA, boxB) is False:
        return 0
    interArea = get_intersection_area(boxA, boxB)
    union = get_union_areas(boxA, boxB, interArea=interArea)
    # intersection over union
    iou = interArea / union
    assert iou >= 0
    return iou

@staticmethod
def boxes_intersect(boxA, boxB):
    """
    Checks if the bounding boxes intersect with each other.

    :param boxA: groundtruth bounding box
    :param boxB: predicted bounding box
    :return: True if intersects, False if not
    """ 
    if boxA[0] > boxB[2]:
        return False  # boxA is right of boxB
    if boxB[0] > boxA[2]:
        return False  # boxA is left of boxB
    if boxA[3] < boxB[1]:
        return False  # boxA is above boxB
    if boxA[1] > boxB[3]:
        return False  # boxA is below boxB
    return True

@staticmethod
def get_intersection_area(boxA, boxB):
    """
    Gets the area of the intersection between the bounding boxes.

    :param boxA: groundtruth bounding box
    :param boxB: predicted bounding box
    :return: area of intersection area
    """ 
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # intersection area
    return (xB - xA + 1) * (yB - yA + 1)

@staticmethod
def get_union_areas(boxA, boxB, interArea=None):
    """
    Gets the area of the union of the bounding boxes.

    :param boxA: groundtruth bounding box
    :param boxB: predicted bounding box
    :param interArea: area of intersection, if it exists (default: None, computes automatically)
    :return: area of union
    """ 
    area_A = get_area(boxA)
    area_B = get_area(boxB)
    if interArea is None:
        interArea = get_intersection_area(boxA, boxB)
    return float(area_A + area_B - interArea)

@staticmethod
def get_area(box):
    """
    Gets the area of the bounding box.

    :param box: bounding box
    :return: area inside the bounding box
    """ 
    return (box[2] - box[0] + 1) * (box[3] - box[1] + 1)

@staticmethod
def calculate_average_precision(rec, prec):
    """
    Calculates the average precision for the different recall and precision values.

    :param rec: list of recall values
    :param prec: list of precision values
    :return: list of the list of average precision, as well as the mean precision and mean recall values
    """ 
    mrec = []
    mrec.append(0)
    [mrec.append(e) for e in rec]
    mrec.append(1)
    mpre = []
    mpre.append(0)
    [mpre.append(e) for e in prec]
    mpre.append(0)
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    ii = []
    for i in range(len(mrec) - 1):
        if mrec[1+i] != mrec[i]:
            ii.append(i + 1)
    ap = 0
    for i in ii:
        ap = ap + np.sum((mrec[i] - mrec[i - 1]) * mpre[i])
    # return [ap, mpre[1:len(mpre)-1], mrec[1:len(mpre)-1], ii]
    return [ap, mpre[0:len(mpre) - 1], mrec[0:len(mpre) - 1], ii]

def calc_metrics(t_labels, t_boxes, p_labels, p_boxes, p_scores, IOUThreshold=0.5):
    """
    Calculates the effectiveness metrics for the predictions for each class using precision, recall, and the average precision

    :param t_labels: list of labels for each groundtruth object in each image
    :param t_boxes: list of bounding boxes for each groundtruth object in each image
    :param p_labels: list of labels for each predicted object in each image
    :param p_boxes: list of bounding boxes for each predicted object in each image
    :param p_scores: list of probability scores for each predicted object in each image
    :param IOUThreshold: threshold used for determining if the IoU calculated indicates a TP or FP (default: 0.5)
    :return: 
        ret: list holding dictionaries for each image, each containing the class as well as the precision, 
        recall, AP, interpolated precision and recall lists, as well as the total positive values, 
        total TP and total FP values
    """ 
    IBEM_CLASSES = getIBEMClasses()
    ret = []  # list containing metrics (precision, recall, average precision) of each class
    # List with all ground truths (Ex: [imageName,class,confidence=1, (bb coordinates XYX2Y2)])
    groundTruths = []
    # List with all detections (Ex: [imageName,class,confidence,(bb coordinates XYX2Y2)])
    detections = []
    # Get all classes
    classes = [i for i in range(len(IBEM_CLASSES))]
    # Loop through all bounding boxes and separate them into GTs and detections
    # [imageName, class, confidence, (bb coordinates XYX2Y2)]
    for i in range(len(t_labels)):
        for j in range(len(t_labels[i])):
            groundTruths.append([
                    i,
                    t_labels[i][j], 1,
                    t_boxes[i][j]
                ])
    for i in range(len(p_labels)):
        for j in range(len(p_labels[i])):
            detections.append([
                    i,
                    p_labels[i][j], 
                    p_scores[i][j],
                    p_boxes[i][j]
                ])
    # Precision x Recall is obtained individually by each class
    # Loop through by classes
    for c in classes:
        # Get only detection of class c
        dects = []
        [dects.append(d) for d in detections if d[1] == c]
        # Get only ground truths of class c, use filename as key
        gts = {}
        npos = 0
        for g in groundTruths:
            if g[1] == c:
                npos += 1
                gts[g[0]] = gts.get(g[0], []) + [g]

        # sort detections by decreasing confidence
        dects = sorted(dects, key=lambda conf: conf[2], reverse=True)
        TP = np.zeros(len(dects))
        FP = np.zeros(len(dects))
        # create dictionary with amount of gts for each image
        det = {key: np.zeros(len(gts[key])) for key in gts}

        # print("Evaluating class: %s (%d detections)" % (str(c), len(dects)))
        # Loop through detections
        for d in range(len(dects)):
            # print('dect %s => %s' % (dects[d][0], dects[d][3],))
            # Find ground truth image
            gt = gts[dects[d][0]] if dects[d][0] in gts else []
            iouMax = sys.float_info.min
            for j in range(len(gt)):
                # print('Ground truth gt => %s' % (gt[j][3],))
                iou = calc_iou(dects[d][3], gt[j][3])
                if iou > iouMax:
                    iouMax = iou
                    jmax = j
            # Assign detection as true positive/don't care/false positive
            if iouMax >= IOUThreshold:
                if det[dects[d][0]][jmax] == 0:
                    TP[d] = 1  # count as true positive
                    det[dects[d][0]][jmax] = 1  # flag as already 'seen'
                    # print("TP")
                else:
                    FP[d] = 1  # count as false positive
                    # print("FP")
            # - A detected "cat" is overlaped with a GT "cat" with IOU >= IOUThreshold.
            else:
                FP[d] = 1  # count as false positive
                # print("FP")
        # compute precision, recall and average precision
        acc_FP = np.cumsum(FP)
        acc_TP = np.cumsum(TP)
        rec = acc_TP / npos
        rec = [0.0 if math.isnan(x) else x for x in rec]
        prec = np.divide(acc_TP, (acc_FP + acc_TP))
        # Depending on the method, call the right implementation
        [ap, mpre, mrec, ii] = calculate_average_precision(rec, prec)
        # add class result in the dictionary to be returned
        r = {
            'class': c,
            'precision': prec,
            'recall': rec,
            'AP': ap,
            'interpolated precision': mpre,
            'interpolated recall': mrec,
            'total positives': npos,
            'total TP': np.sum(TP),
            'total FP': np.sum(FP)
        }
        ret.append(r)
    return ret
    
@torch.no_grad()
def predict(images, model, detection_threshold=0.5):
    """
    Predict the output of images after forward pass through
    the model and return the bounding boxes, class names, and 
    class labels. Only returns output with a probability score 
    higher than a set detection threshold.

    :param images: images to pass through the model
    :param model: FCOS model
    :param detection_threshold: threshold for probability score (default: 0.5)
    :return: 
        pred_boxes: predicted bounding boxes for the images
        pred_labels: predicted labels for the images
        pred_scores: predicted probability scores for the images
    """
    # b_s_time = time.time()

    outputs = None
    pred_labels = []
    pred_boxes = []
    pred_scores = []

    # Get the predictions on the image.
    with torch.no_grad():
        outputs = model(images) 

    for j in range(len(outputs)):
        # Get score for all the predicted objects.
        scores = outputs[j]['scores'].detach().cpu().numpy()
        # pred_scores.append(scores[scores >= detection_threshold])
        pred_scores.append(scores)
        bboxes = outputs[j]['boxes'].detach().cpu().numpy()
        # pred_boxes.append(bboxes[scores >= detection_threshold].astype(np.int32))
        pred_boxes.append(bboxes.astype(np.int32))
        labels = outputs[j]['labels'].detach().cpu().numpy()
        # pred_labels.append(labels[scores >= detection_threshold].astype(np.int64))
        pred_labels.append(labels.astype(np.int64))

    # p_fps = batch_size/(time.time() - b_s_time)
    # print("Test batch", p_fps)

    return pred_boxes, pred_labels, pred_scores

@torch.no_grad()
def validate(loader, model, device, result_dir=None):
    """
    Runs through a set of data for prediction/validation and calculates the 
    Mean Average Precision (mAP), overall precision, and overall recall. 
    If specified, stores images with bounding boxes and labels drawn on them 
    in a results directory.

    :param loader: dataloader containing images to pass through the model
    :param model: FCOS model
    :param device: run on cpu or cuda
    :param result_dir: path to directory to store images with labels and bounding boxes 
    drawn on them (default: None, does not put images in results directory)
    :return: 
        mAP: mean average precision score
        mean_prec: mean precision for the images
        mean_rec: mean recall for the images
    """
    IBEM_CLASSES = getIBEMClasses()
    torch.cuda.empty_cache()
    # timer = time.time()
    model = set_eval_model(model, device)
    # print('Set Eval Mode:',time.time() - timer)
    precision = []
    recall = []
    ap = []
    i_prec = []
    i_rec = []
    for i, batch in enumerate(loader):
        # timer = time.time()
        img_list, target_list = batch

        boxes, labels, scores = predict(img_list, model, 0.5)
        # print("Boxes:", boxes)
        # print("Labels:", labels)

        label_list = []
        box_list = []
        for target in target_list:
            label_list.append(target['labels'].detach().cpu().numpy())
            box_list.append(target['boxes'].detach().cpu().numpy())
            
        metrics = calc_metrics(label_list, box_list, labels, boxes, scores)
        for metric in metrics:
            # print('interpolated precision:', metric['interpolated precision'])
            # print('interpolated recall:', metric['interpolated recall'])
            # print('precision:', metric['precision'])
            # print('recall:', metric['recall'])
            # print('AP', metric['AP'])
            t_prec = np.mean(metric['interpolated precision'])
            t_rec = np.mean(metric['interpolated recall'])
            precision.append(t_prec)
            recall.append(t_rec)
            ap.append(metric['AP'])
            i_prec.append(metric['interpolated precision'])
            i_rec.append(metric['interpolated recall'])

        if result_dir is not None:
            for j in range(len(img_list)):
                c_array = img_list[j].cpu().numpy().transpose(1, 2, 0) #convert from RGB to BGR
                # print(c_array)
                # print(test_dataset[0][0])
                # out_img = test_dataset[0][0].to("cpu")#.permute(1, 2, 0).to("cpu")
                # print(out_img)
                # out_img = transforms.ToPILImage()(out_img)
                # # plt.imshow(out_img)
                # print(out_img)
                # out_img.save('test_img_pred.jpg')
                image = cv2.cvtColor(c_array, cv2.COLOR_RGB2BGR)
                image = cv2.normalize(image, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
                if len(boxes[j] > 0):
                    image = draw_boxes(boxes[j], labels[j], image)
                save_name = 'batch' + str(i) + '_img' + str(j) + '.jpg'
                # print(image)
                # cv2.imwrite(os.path.join(result_dir, save_name), image) #image[:, :, ::-1]
    mAP = np.mean(ap)
    mean_prec = np.mean(precision)
    mean_rec = np.mean(recall)
    return mAP, mean_prec, mean_rec, i_prec, i_rec