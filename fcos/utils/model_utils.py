"""
File: model_utils.py
Author: Angel Wheelwright
Date: 2023-24-10
Description: Contains utility functions for creating the FCOS detection model and setting modes for the model
Uses code from:
https://learnopencv.com/fcos-anchor-free-object-detection-explained/
"""
import torchvision

def get_model(device, num_classes, score_thresh=0.05):    
    """
    Function to initialize and return a new FCOS model with a FPN and Resnet50 backbone for the purpose of detecting objects in images, 
    classifying them, and forming bounding boxes around them.

    :param device: run on cpu or cuda
    :param num_classes: number of possible classes included as part of the dataset, including background (default: 4)
    :return: 
        model: FCOS model
    """ 
    model = torchvision.models.detection.fcos_resnet50_fpn(progress=False, num_classes=num_classes).to(device)
    model.score_thresh = score_thresh
    model.max_size = 2048
    return model

def set_eval_model(model, device):
    """
    Loads the detection model onto the given computation device and sets the model to evaluation mode.

    :param model: FCOS model
    :param device: run on cpu or cuda
    :return: 
        model: FCOS model
    """ 
    model = model.eval().to(device)
    return model

def set_train_model(model, device):
    """
    Loads the detection model onto the given computation device and sets the model to training mode.

    :param model: FCOS model
    :param device: run on cpu or cuda
    :return: 
        model: FCOS model
    """ 
    model = model.train().to(device)
    return model
