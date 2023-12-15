"""
File: data_utils.py
Author: Angel Wheelwright
Date: 2023-24-10
Description: Contains utility functions for dealing with data, particularly a custom IBEM dataset class, a custom dataloader function, and more global data and transformation functions
"""

import torch
import cv2
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import Dataset
import json
import os
import numpy as np

# Mathematical formula class types in the dataset.
def getIBEMClasses():
    IBEM_CLASSES = ['__background__', 
    'isolated', 
    'embedded', 
    'embedded_split'
    ]
    return IBEM_CLASSES

# Define the torchvision image transforms.
transform = transforms.Compose([
    transforms.ToTensor()
])

class IBEMDataset(Dataset):
    """IBEM dataset."""

    def __init__(self, lst_file, json_file, img_dir, device):
        """
        Calculates the average precision for the different recall and precision values.

        :param lst_file: path to lst_file with image names to reference for dataset
        :param json_file: path to the json file with labels and bounding box annotations
        :param image_dir: path to directory where the images are stored
        :param device: run on cpu or cuda
        """ 
        if type(lst_file) is list:
            self.lst_file = []
            for item in lst_file:
                temp_list = []
                with open(item, 'r') as f:
                    temp_list = f.read().splitlines()
                if temp_list[-1] is None:
                    temp_list.pop()
                self.lst_file += temp_list
        else:
            temp_list = []
            with open(lst_file, 'r') as f:
                temp_list = f.read().splitlines()
            if temp_list[-1] is None:
                temp_list.pop()
            self.lst_file = temp_list
        self.json_file = convert_json(json_file)
        self.img_dir = img_dir
        self.device = device
        # print(self.lst_file)
    def __len__(self):
        """
        Calculates number of entries in the dataset.

        :return: length of the retrieved image set
        """ 
        return len(self.lst_file)

    def __getitem__(self, idx):
        """
        Returns the image and annotations for that image at a specified position in the dataset.

        :param idx: index of image data to retrieve
        :return: [image, dictionary containing bounding boxes and labels for that image]
        """ 
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_name = self.lst_file[idx]
        rem_index = img_name.index('e') + 1
        t_img_name = img_name[:rem_index] + img_name[rem_index+1:]
        img_path = self.img_dir + t_img_name + '.jpg'

        img = cv2.imread(img_path)
        image = transform(img).to(self.device)

        c_dict = self.json_file[img_name]

        b_box_list = []
        label_list = []

        if 'isolated' in c_dict.keys():
            i_dict = c_dict['isolated']
            for i in range(len(i_dict.keys())):
                # print(image.shape)
                # print(i_dict[str(i)]['x_min'], i_dict[str(i)]['y_min'], i_dict[str(i)]['x_max'], i_dict[str(i)]['y_max'])
                box_x_min = (image.shape[2] / 100.0) * i_dict[str(i)]['x_min']
                box_y_min = (image.shape[1] / 100.0) * i_dict[str(i)]['y_min']
                box_x_max = (image.shape[2] / 100.0) * i_dict[str(i)]['x_max']
                box_y_max = (image.shape[1] / 100.0) * i_dict[str(i)]['y_max']
                b_box = [box_x_min, box_y_min, box_x_max, box_y_max]
                # print("b_box:", b_box)
                b_box_list.append(b_box)
                label_list.append(1) # isolated
        if 'embedded' in c_dict.keys():
            e_dict = c_dict['embedded']
            for i in range(len(e_dict.keys())):
                box_x_min = (image.shape[2] / 100.0) * e_dict[str(i)]['x_min']
                box_y_min = (image.shape[1] / 100.0) * e_dict[str(i)]['y_min']
                box_x_max = (image.shape[2] / 100.0) * e_dict[str(i)]['x_max']
                box_y_max = (image.shape[1] / 100.0) * e_dict[str(i)]['y_max']
                b_box = [box_x_min, box_y_min, box_x_max, box_y_max]
                b_box_list.append(b_box)
                # label_list.append(2) # embedded
                if e_dict[str(i)]['split'] == 'single':
                    label_list.append(2) # embedded
                else:
                    label_list.append(3) # embedded_split

        label_dict = {'boxes': torch.FloatTensor(b_box_list).to(self.device), 'labels': torch.tensor(label_list, dtype=torch.int64).to(self.device)}
        return [image, label_dict]

def has_only_empty_bbox(annot):
    return all(any(o <= 1 for o in obj['bbox'][2:]) for obj in annot)


def has_valid_annotation(annot):
    if len(annot) == 0:
        return False

    if has_only_empty_bbox(annot):
        return False

    return True

class COCODataset(datasets.CocoDetection):
    def __init__(self, path, split, device, transform=None):
        root = os.path.join(path, f'{split}2017')
        annot = os.path.join(path, 'annotations', f'instances_{split}2017.json')

        super().__init__(root, annot)

        self.ids = sorted(self.ids)

        if split == 'train':
            ids = []

            for id in self.ids:
                ann_ids = self.coco.getAnnIds(imgIds=id, iscrowd=None)
                annot = self.coco.loadAnns(ann_ids)

                if has_valid_annotation(annot):
                    ids.append(id)

            self.ids = ids

        # print(annot)
        self.category2id = {v: i + 1 for i, v in enumerate(self.coco.getCatIds())}
        self.id2category = {v: k for k, v in self.category2id.items()}
        self.id2img = {k: v for k, v in enumerate(self.ids)}

        self.transform = transform
        self.device = device

    def __getitem__(self, index):
        img, annot = super().__getitem__(index)
        cv2_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        image = transform(cv2_img).to(self.device)

        annot = [o for o in annot if o['iscrowd'] == 0]

        boxes = [o['bbox'] for o in annot]
        for b in range(len(boxes)):
            width = boxes[b][2]
            height = boxes[b][3]
            if width < 1:
                # print("Width < 1")
                width += 1
            if height < 1:
                # print("Height < 1")
                height += 1
            boxes[b] = [boxes[b][0], boxes[b][1], boxes[b][0] + width, boxes[b][1] + height]

        # boxes = torch.as_tensor(boxes).reshape(-1, 4)

        classes = [o['category_id'] for o in annot]
        classes = [self.category2id[c] for c in classes]
        # classes = torch.tensor(classes)
        # print(min(classes), max(classes))
        # classes = [self.category2id[c] for c in labels]
        # classes = torch.tensor(classes)

        label_dict = {'boxes': torch.FloatTensor(boxes).to(self.device), 'labels': torch.tensor(classes, dtype=torch.int64).to(self.device)}
        return [image, label_dict] 

    def get_image_meta(self, index):
        id = self.id2img[index]
        img_data = self.coco.imgs[id]

        return img_data

def loader_collate(batch):
    """
    Method used by a dataloader to take a batch of images and annotations, save them as a image list and annotation list,
    and return them when iterating through the dataloader.

    :param batch: a list of tuples containing an image and the annotations for that image
    :return: 
        img_list: list of images
        anno_list: list of annotations
    """ 
    img_list = []
    anno_list = []
    for tuple in batch:
        img_list.append(tuple[0])
        anno_list.append(tuple[1])
    return img_list, anno_list

def convert_json(path):
    """
    Opens .json file from a path, then converts the data to a json dictionary.

    :param path: file path for the .json file
    :return: 
        data: json data contained in the file
    """ 
    file = open(path, 'r')
    data = json.load(file)
    file.close()
    return data

# path = 'IBEM/partitions/a_c/Va01_c.lst'
# path_mod = 'IBEM/partitions/Va01_cp.lst'

def check_partitions(path, path_mod, j_file, img_dir):
    """
    Opens .json file from a path, then converts the data to a json dictionary.

    :param path: file path for the .json file
    :return: 
        data: json data contained in the file
    """ 
    lst = []
    with open(path, 'r') as f:
        lst = f.read().splitlines()
    if lst[-1] is None:
        lst.pop()
    dict_j = convert_json(j_file)
    with open(path_mod, 'w+') as f:
        for item in lst:
            if item in dict_j.keys():
                f.write('%s\n' %item)
    print("File written successfully")
    f.close()

def check_page(path, path_mod, img_dir):
    lst = []
    with open(path, 'r') as f:
        lst = f.read().splitlines() 
    if lst[-1] is None:
        lst.pop()
    with open(path_mod, 'w+') as f:
        for item in lst:
            rem_index = item.index('e') + 1
            t_img_name = item[:rem_index] + item[rem_index+1:]
            img_path = img_dir + t_img_name + '.jpg'

            # image = torchvision.io.read_image(img_path)
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                if img.any():
                    f.write('%s\n' %item)
    print("File written successfully")
    f.close()

def download_file(url, save_name):
    url = url
    if not os.path.exists(save_name):
        file = requests.get(url)
        open(save_name, 'wb').write(file.content)

# Unzip the data file
def unzip(zip_file=None):
    try:
        with zipfile.ZipFile(zip_file) as z:
            z.extractall("./")
            print("Extracted all")
    except:
        print("Invalid file")