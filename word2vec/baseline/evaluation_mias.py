import os
from collections import OrderedDict
import math
def evaluate():
    p_1 = 0
    p_3 = 0
    p_5=0
    count = 0
    p_3_count =0
    p_5_count =0
    # valid_files = []
    # with open ("../valid_file.txt", 'r') as vf:
    #     valid_files = vf.readlines()
    # valid_files = [''.join(x[:-1]) for x in valid_files]
    # print(valid_files)
    for f in os.listdir('mias_result/'):
        # if f.split('.')[0] in valid_files:
        if True:
            with open('mias_result/'+f, 'r') as rf:
                content_list =rf.readlines()
            if content_list:
                count += 1

            eva = {}
            number = 0
            for rank, line in enumerate(content_list):
                fileName = line.split('/')[-1].split('.')[0]
                if fileName not in eva.keys():
                    eva[fileName] = number
                    number+=1

            judge = []
            origin_file_count = 0
            for dataset_file in os.listdir("../../data/MSE_dataset_full/dataset_full/text/" + f.split('.')[0] + "/answers"):
                cleaned_name = dataset_file.split('.')[0]
                origin_file_count+=1
                if True:
                    judge.append(cleaned_name)
            judge.sort(key=lambda x: int(x.split("_")[1]), reverse=True)

            if origin_file_count >=3:
                p_3_count+=3
            else:
                p_3_count +=origin_file_count
            if origin_file_count >=5:
                p_5_count+=5
            else:
                p_5_count+=origin_file_count

            if judge[0] in eva.keys():
                if (eva[judge[0]]==0):
                    p_1+=1
            for j in judge[:3]:
                if j in eva.keys() and eva[j]<3:
                    p_3+=1
            for j5 in judge[:5]:
                if j5 in eva.keys() and eva[j5] < 5:
                    p_5+=1

    print(count)
    print("P@1: ",p_1/count)
    print("p@3: ", p_3/p_3_count)
    print("p@5: ", p_5/p_5_count)
    # with open("../evaluation_result_tangent.txt", "w") as ev:
    with open("evaluation_result_mias_1.txt", "w") as ev:
        ev.write("p@1: "+str(p_1/count)+"\n")
        ev.write("p@3: "+str(p_3/p_3_count)+"\n")
        ev.write("p@5: "+str(p_5/p_5_count)+"\n")

def MRR():
    count = 0
    score = 0
    # valid_files = []
    # with open("../valid_file.txt", 'r') as vf:
    #     valid_files = vf.readlines()
    # valid_files = [''.join(x[:-1]) for x in valid_files]
    # print(valid_files)
    for f in os.listdir('mias_result/'):
        # if f.split('.')[0] in valid_files:
        if True:
            with open('mias_result/' + f, 'r') as rf:
                content_list = rf.readlines()
            if content_list:
                count += 1

            eva = {}
            number = 0
            for rank, line in enumerate(content_list):
                fileName = line.split('/')[-1].split('.')[0]
                if fileName not in eva.keys():
                    eva[number] = fileName
                    number += 1

            judge = []
            origin_file_count = 0
            for dataset_file in os.listdir("../../data/MSE_dataset_full/dataset_full/text/" + f.split('.')[0] + "/answers"):
                cleaned_name = dataset_file.split('.')[0]
                origin_file_count += 1
                if True:
                    judge.append(cleaned_name)
            judge.sort(key=lambda x: int(x.split("_")[1]), reverse=True)
            position = judge.index(eva[0])
            # print(position)
            current_score = 1 / (position + 1)
            score += current_score
            with open("mrr_mias.txt", 'a') as mrrf:
                mrrf.write(str(current_score) + ",")
    mrr = (1 / count) * score
    print(mrr)

def nDCG():
    count = 0
    score = 0
    # valid_files = []
    # with open("../valid_file.txt", 'r') as vf:
    #     valid_files = vf.readlines()
    # valid_files = [''.join(x[:-1]) for x in valid_files]
    # print(valid_files)
    for f in os.listdir('mias_result/'):
        # if f.split('.')[0] in valid_files:
        if True:
            with open('mias_result/' + f, 'r') as rf:
                content_list = rf.readlines()
            if content_list:
                count += 1

            eva = []
            number = 0
            for rank, line in enumerate(content_list):
                fileName = line.split('/')[-1].split('.')[0]
                if fileName not in eva:
                    eva.append(fileName)
                    number += 1

            judge = []
            origin_file_count = 0
            for dataset_file in os.listdir("../../data/MSE_dataset_full/dataset_full/text/" + f.split('.')[0] + "/answers"):
                cleaned_name = dataset_file.split('.')[0]
                origin_file_count += 1
                if True:
                    judge.append(cleaned_name)
            judge.sort(key=lambda x: int(x.split("_")[1]), reverse=True)
            dcg = 5 - judge.index(eva[0])
            idcg = 5 / math.log(2, 2) + 4 / math.log(3, 2) + 3 / math.log(4, 2) + 2 / math.log(5, 2) + 1 / math.log(6,
                                                                                                                    2)
            for i in range(1, 5):
                if i > len(eva) - 1:
                    cg = 0
                else:
                    cg = 5 - judge.index(eva[i])
                discount = math.log(i + 1 + 1, 2)
                dcg += cg / discount
            ndcg = dcg / idcg
            score += ndcg
            with open("ndcg_mias.txt", 'a') as ndcgf:
                ndcgf.write(str(ndcg) + ",")
    print(score / count)
if __name__ == "__main__":
    MRR()