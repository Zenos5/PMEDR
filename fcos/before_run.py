from transformers import GPT2Tokenizer
from utils.data_utils import download_file, unzip
from utils.data_utils import IBEMDataset, COCODataset, loader_collate
from torch.utils.data import DataLoader
import cv2
import os
import numpy as np
from utils.display_utils import draw_boxes

if __name__ == "__main__":
    # tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    # tokenizer.save_pretrained("./tokenizer/")
    # download_file(
    #     'https://www.dropbox.com/s/ukc7wocsn7xrm2r/data.zip?dl=1',
    #     'data.zip'
    # )
    # unzip('data.zip')
    device = "cpu"

    j_file = 'IBEM/IBEM.json'
    i_dir = 'IBEM/pages/'

    train_lst = ['IBEM/partitions/a_cp/Tr00_cp.lst',
        'IBEM/partitions/a_cp/Tr01_cp.lst',
        'IBEM/partitions/a_cp/Tr10_cp.lst',
        'IBEM/partitions/a_cp/Ts00_cp.lst',
        'IBEM/partitions/a_cp/Ts01_cp.lst',
        'IBEM/partitions/a_cp/Ts10_cp.lst',
        'IBEM/partitions/a_cp/Va00_cp.lst'
    ]

    train_dataset = IBEMDataset(lst_file=train_lst, json_file=j_file, img_dir=i_dir, device=device)
    val_dataset = IBEMDataset(lst_file=['IBEM/partitions/a_cp/Va01_cp.lst'], json_file=j_file, img_dir=i_dir, device=device)

    print("Size of IBEM train dataset:", len(train_dataset))
    print("Size of IBEM val dataset:", len(val_dataset))

    # print("First training element:")
    # print(train_dataset[0])
    # print(train_dataset[0][0].shape)
    
    train_loader = DataLoader(train_dataset, batch_size=6, shuffle=True, collate_fn=loader_collate)

    data_iter = iter(train_loader)
    batch = next(data_iter)
    img_list, target_list = batch
    # print("Image List:")
    # print(img_list)
    # print("Target List:")
    # print(target_list)

    c_array = img_list[0].cpu().numpy().transpose(1, 2, 0) #convert from RGB to BGR
    image = cv2.cvtColor(c_array, cv2.COLOR_RGB2BGR)
    image = cv2.normalize(image, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
    
    print(image.shape)
    # boxes = target_list[0]['boxes']
    # for box in boxes:
    #     box[0] = (image.shape[1] / 100.0) * box[0]
    #     box[1] = (image.shape[0] / 100.0) * box[1]
    #     box[2] = (image.shape[1] / 100.0) * box[2]
    #     box[3] = (image.shape[0] / 100.0) * box[3]
    # print('Boxes:',boxes)
    if len(target_list[0]['boxes'] > 0):
        image = draw_boxes(target_list[0]['boxes'], target_list[0]['labels'], image)
    save_name = 'img' + str(0) + '.jpg'
    # print(image)
    result_dir = "results"
    cv2.imwrite(os.path.join(result_dir, save_name), image)



    # train_dataset = COCODataset("COCO", "train", device)
    # val_dataset = COCODataset("COCO", "val", device)
    # print("Size of COCO train dataset:", len(train_dataset))
    # print("Size of COCO val dataset:", len(val_dataset))

    # print("First training element:")
    # print(train_dataset[0])
    # print(train_dataset[0][0].shape)

    # train_loader = DataLoader(train_dataset, batch_size=6, shuffle=True, collate_fn=loader_collate)

    # data_iter = iter(train_loader)
    # batch = next(data_iter)
    # img_list, target_list = batch
    # print("Image List:")
    # print(img_list)
    # print("Target List:")
    # print(target_list)

