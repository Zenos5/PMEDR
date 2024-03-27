import torch
import cv2
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import json
import os
import argparse
from utils.data_utils import convert_json, check_page
import math

def extract_data(partition, lst_path, data_dir):
    lst_file = []
    with open(data_dir + lst_path, 'r') as f:
        lst_file = f.read().splitlines()
    print(lst_file[-1])
    if lst_file[-1] is None:
        lst_file.pop()
    json_file = convert_json(data_dir + 'IBEM.json')
    img_dir = data_dir + 'pages/'
    latex_dict = {}
    
    save_path_bbmi = data_dir + 'bb/' + partition + 'images/'
    save_path_bbml = data_dir + 'bb/' + partition
    os.makedirs(data_dir + 'bb/', exist_ok=True)
    os.makedirs(save_path_bbml, exist_ok=True)
    os.makedirs(save_path_bbmi, exist_ok=True)

    for idx in range(len(lst_file)):
        img_name = lst_file[idx]
        rem_index = img_name.index('e') + 1
        t_img_name = img_name[:rem_index] + img_name[rem_index+1:]
        img_path = img_dir + t_img_name + '.jpg'
        img = cv2.imread(img_path)
        c_dict = json_file[img_name]
        # print(img.shape)
        
        if 'isolated' in c_dict.keys():
            i_dict = c_dict['isolated']
            for i in range(len(i_dict.keys())):
                box_x_min = (img.shape[1] / 100.0) * i_dict[str(i)]['x_min']
                box_y_min = (img.shape[0] / 100.0) * i_dict[str(i)]['y_min']
                box_x_max = (img.shape[1] / 100.0) * i_dict[str(i)]['x_max']
                box_y_max = (img.shape[0] / 100.0) * i_dict[str(i)]['y_max']
                mod_img = img[math.floor(box_y_min):math.ceil(box_y_max), math.floor(box_x_min):math.ceil(box_x_max)]
                img_name = str(idx) + '_isolated_' + str(i) + '.jpg'
                img_path = save_path_bbmi + img_name
                if mod_img.any():
                    cv2.imwrite(img_path, mod_img)
                    latex_dict[img_name] = i_dict[str(i)]['latex_norm']
        if 'embedded' in c_dict.keys():
            e_dict = c_dict['embedded']
            for i in range(len(e_dict.keys())):
                box_x_min = (img.shape[1] / 100.0) * e_dict[str(i)]['x_min']
                box_y_min = (img.shape[0] / 100.0) * e_dict[str(i)]['y_min']
                box_x_max = (img.shape[1] / 100.0) * e_dict[str(i)]['x_max']
                box_y_max = (img.shape[0] / 100.0) * e_dict[str(i)]['y_max']
                mod_img = img[math.floor(box_y_min):math.ceil(box_y_max), math.floor(box_x_min):math.ceil(box_x_max)]
                img_name = str(idx) + '_embedded_' + str(i) + '.jpg'
                img_path = save_path_bbmi + img_name
                if mod_img.any():
                    cv2.imwrite(img_path, mod_img)
                    latex_dict[img_name] = e_dict[str(i)]['latex_norm']
    dict_path = save_path_bbml + 'latex.json'
    with open(dict_path, "w") as outfile:
        json.dump(latex_dict, outfile)

if __name__ == "__main__":
    print("Tr00/")
    data_dir = "../data/IBEM/"
    extract_data('Tr00/', 'partitions/a_cp/Tr00_cp.lst', data_dir)
    # check_page('IBEM/partitions/a_cp/Tr00_cp.lst', 'IBEM/partitions/a_cpf/Tr00_cp.lst', 'IBEM/pages/')
    print("Tr01/")
    extract_data('Tr01/', 'partitions/a_cp/Tr01_cp.lst', data_dir)
    # check_page('IBEM/partitions/a_cp/Tr01_cp.lst', 'IBEM/partitions/a_cpf/Tr01_cp.lst', 'IBEM/pages/')
    print("Tr10/")
    extract_data('Tr10/', 'partitions/a_cp/Tr10_cp.lst', data_dir)
    # check_page('IBEM/partitions/a_cp/Tr10_cp.lst', 'IBEM/partitions/a_cpf/Tr10_cp.lst', 'IBEM/pages/')
    print("Ts01/")
    extract_data('Ts01/', 'partitions/a_cp/Ts01_cp.lst', data_dir)
    # check_page('IBEM/partitions/a_cp/Ts01_cp.lst', 'IBEM/partitions/a_cpf/Ts01_cp.lst', 'IBEM/pages/')
    print("Ts00/")
    extract_data('Ts00/', 'partitions/a_cp/Ts00_cp.lst', data_dir)
    # check_page('IBEM/partitions/a_cp/Ts00_cp.lst', 'IBEM/partitions/a_cpf/Ts00_cp.lst', 'IBEM/pages/')
    print("Ts11/")
    extract_data('Ts11/', 'partitions/a_cp/Ts11_cp.lst', data_dir)
    # check_page('IBEM/partitions/a_cp/Ts11_cp.lst', 'IBEM/partitions/a_cpf/Ts11_cp.lst', 'IBEM/pages/')
    print("Ts10/")
    extract_data('Ts10/', 'partitions/a_cp/Ts10_cp.lst', data_dir)
    # check_page('IBEM/partitions/a_cp/Ts10_cp.lst', 'IBEM/partitions/a_cpf/Ts10_cp.lst', 'IBEM/pages/')
    print("Va01/")
    extract_data('Va01/', 'partitions/a_cp/Va01_cp.lst', data_dir)
    # check_page('IBEM/partitions/a_cp/Va01_cp.lst', 'IBEM/partitions/a_cpf/Va01_cp.lst', 'IBEM/pages/')
    print("Va00/")
    extract_data('Va00/', 'partitions/a_cp/Va00_cp.lst', data_dir)
    # check_page('IBEM/partitions/a_cp/Va00_cp.lst', 'IBEM/partitions/a_cpf/Va00_cp.lst', 'IBEM/pages/')