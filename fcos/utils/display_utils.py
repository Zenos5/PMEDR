"""
File: display_utils.py
Author: Angel Wheelwright
Date: 2023-24-10
Description: Contains utility functions for displaying data
Uses code from:
https://learnopencv.com/fcos-anchor-free-object-detection-explained/
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
import os
from utils.data_utils import getIBEMClasses

# Create different colors for each class.
np.random.seed(42)
COLORS = np.random.uniform(0, 255, size=(len(getIBEMClasses()), 3))

# Plot and visualize images in a 2x2 grid.
def visualize(result_dir):
    """
    Function accepts a list of images and plots them in a 2x2 grid.

    :param result_dir: directory of images
    """
    plt.figure(figsize=(20, 18))
    image_names = glob.glob(os.path.join(result_dir, '*.jpg'))
    for i, image_name in enumerate(image_names):
        image = plt.imread(image_name)
        plt.subplot(2, 2, i+1)
        plt.imshow(image)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Plot result losses/scores per epoch
def plot_results(results, data_type, title):
    """
    Function that plots results per epoch on a graph.

    :param results: list of results (ex: scores, losses)
    :param data_type: label for the y-axis (what kind of result is it)
    :param title: title for the plot
    """
    plt.plot(results)
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel(data_type)
    plt.legend()
    plt.show()

def draw_boxes(boxes, labels, image):
    """
    Draws the bounding box around a detected object.

    :param boxes: bounding boxes
    :param labels: labels of objects in the image
    :param image: original image
    :return: image with bounding boxes drawn on and labeled
    """
    IBEM_CLASSES = getIBEMClasses()

    lw = max(round(sum(image.shape) / 2 * 0.003), 2) # Line width.
    tf = max(lw - 1, 1) # Font thickness.
    
    # print(image.shape)
    

    for i, box in enumerate(boxes):
        # print(box)
        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        color = COLORS[labels[i]]
        class_name = IBEM_CLASSES[labels[i]]
        cv2.rectangle(
            image,
            p1,
            p2,
            color[::-1],
            thickness=lw,
            lineType=cv2.LINE_AA
        )
        # For filled rectangle.
        w, h = cv2.getTextSize(
            class_name, 
            0, 
            fontScale=lw / 3, 
            thickness=tf
        )[0]  # Text width, height
        
        outside = p1[1] - h >= 3
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        
        cv2.rectangle(
            image, 
            p1, 
            p2, 
            color=color[::-1], 
            thickness=-1, 
            lineType=cv2.LINE_AA
        )  
        cv2.putText(
            image, 
            class_name, 
            (p1[0], p1[1] - 5 if outside else p1[1] + h + 2),
            cv2.FONT_HERSHEY_SIMPLEX, 
            fontScale=lw / 3.8, 
            color=(255, 255, 255), 
            thickness=tf, 
            lineType=cv2.LINE_AA
        )
    return image