import torch
from data_utils import IBEMDataset, getIBEMClasses
from model_utils import get_model
import json

if __name__ == "__main__":
    # j_file = '../IBEM/IBEM.json'
    # i_dir = '../IBEM/pages/'
    # IBEM_CLASSES = getIBEMClasses()
    # train_dataset = IBEMDataset(lst_file=['../IBEM/partitions/a_cp/Tr00_cp.lst', '../IBEM/partitions/a_cp/Tr01_cp.lst'], json_file=j_file, img_dir=i_dir, device='cpu')
    # model = get_model(device='cpu', num_classes=len(IBEM_CLASSES))
    # model.score_thresh = 0.05
    # print(model.score_thresh)
    # print(len(train_dataset))

    val_i_rec = [[1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0]]

    val_i_prec = []

    # with open('test.txt', 'w') as fp:
    #     json.dump(val_i_rec, fp)
    
    with open('test.txt', 'r') as fp:
        val_i_prec = json.load(fp)
    print(val_i_prec)
    print(val_i_prec[0][0] + val_i_prec[0][1])
    print(type(val_i_prec[0][0]))
