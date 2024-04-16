import os
from collections import OrderedDict
def evaluate():

    for f in os.listdir('lda_15_3_4/combined'):
        with open('RESULT/combined/'+f, 'r') as rf:
            content_list_bm =rf.readlines()
        eva_bm = {}
        for rank, line in enumerate(content_list_bm):
            fileName = line.split(',')[0].replace('\'', '').replace('(', '')
            eva_bm[fileName] = rank
        with open('lda_15_3_4/combined/'+f, 'r') as rf:
            content_list_ti =rf.readlines()
        eva_ti = {}
        for rank, line in enumerate(content_list_ti):
            fileName = line.split(',')[0].replace('\'', '').replace('(', '')
            eva_ti[fileName] = rank

        judge = []
        origin_file_count = 0
        for dataset_file in os.listdir("../../data/MSE_dataset_full/dataset_full/text/" + f.split('.')[0] + "/answers"):
            cleaned_name = dataset_file.split('.')[0]
            origin_file_count+=1
            if cleaned_name in eva_bm.keys():
                judge.append(cleaned_name)
        judge.sort(key=lambda x: int(x.split("_")[1]), reverse=True)

        p_3_count = 0
        p_5_count = 0

        if origin_file_count >=3:
            p_3_count=3
        else:
            p_3_count +=origin_file_count
        if origin_file_count >=5:
            p_5_count=5
        else:
            p_5_count+=origin_file_count

        with open("p_1_wix_b_l.txt", "a") as f1b:
            if eva_ti[judge[0]] == 0:
                f1b.write(str(1))
            else:
                f1b.write(str(0))

        with open("p_1_wix_b_l.txt", "a") as f1t:
            if eva_bm[judge[0]] == 0:
                f1t.write(","+str(1)+"\n")
            else:
                f1t.write(","+str(0)+"\n")

        # count_3 = 0
        # for j in judge[:3]:
        #     if eva_bm[j] < p_3_count:
        #         count_3 += 1
        # with open("p_3_wix.txt", "a") as f3b:
        #     f3b.write(str(count_3))
        # count_3 = 0
        # for j in judge[:3]:
        #     if eva_ti[j] < p_3_count:
        #         count_3 += 1
        # with open("p_3_wix.txt", "a") as f3t:
        #     f3t.write(","+str(count_3)+"\n")
        #
        # count_5 = 0
        # for j in judge[:5]:
        #     if eva_bm[j] < p_5_count:
        #         count_5 += 1
        # with open("p_5_wix.txt", "a") as f5b:
        #     f5b.write(str(count_5))
        # count_5 = 0
        # for j in judge[:5]:
        #     if eva_ti[j] < p_5_count:
        #         count_5 += 1
        # with open("p_5_wix.txt", "a") as f5t:
        #     f5t.write("," + str(count_5) + "\n")


if __name__ == "__main__":
    evaluate()