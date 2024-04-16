import os
from collections import OrderedDict


import os
from collections import OrderedDict
def evaluate():
    p_1 = 0
    p_3 = 0
    p_5=0
    count = 0
    p_3_count =0
    p_5_count =0
    for f in os.listdir('../cft_result/'):
        with open('../cft_result/'+f, 'r') as rf:
            content_list_cft =rf.readlines()
        content_list_lda=[]
        isExist = os.path.exists('../RESULT_tfidf/combined/' + f+".txt")
        if isExist:
            with open('../RESULT_tfidf/combined/' + f+".txt", 'r') as rf:
                content_list_lda = rf.readlines()
        eva_cft = {}
        if content_list_cft and content_list_lda:
            count += 1
            for rank, line in enumerate(content_list_cft):
                fileName = line.split(' ')[2].split('_')[0]
                if fileName not in eva_cft.keys():
                    eva_cft[fileName] = line.split(' ')[3]

            eva_lda = {}
            for rank, line in enumerate(content_list_lda):
                fileName = line.split(',')[0].replace('\'', '').replace('(', '').split("_")[0]
                eva_lda[fileName] = rank
            judge = []
            origin_file_count = 0
            for dataset_file in os.listdir("../../data/MSE_dataset_full/dataset_full/text/" + f.split('.')[0] + "/answers"):
                cleaned_name = dataset_file.split('.')[0]
                origin_file_count+=1
                if True:
                    judge.append(cleaned_name)
            judge.sort(key=lambda x: int(x.split("_")[1]), reverse=True)
            judge = [j.split('_')[0] for j in judge]

            p_3_count = 0
            p_5_count = 0

            if origin_file_count >= 3:
                p_3_count = 3
            else:
                p_3_count += origin_file_count
            if origin_file_count >= 5:
                p_5_count = 5
            else:
                p_5_count += origin_file_count

            with open("../wilcoxon_result/p_1_wix_cft_tfidf.txt", "a") as f1b:
                if (judge[0] in eva_cft.keys()) and int(eva_cft[judge[0]]) == 1:
                    f1b.write(str(1))
                else:
                    f1b.write(str(0))

            with open("../wilcoxon_result/p_1_wix_cft_tfidf.txt", "a") as f1t:
                if (judge[0] in eva_lda.keys()) and int(eva_lda[judge[0]]) == 0:
                    f1t.write("," + str(1) + "\n")
                else:
                    f1t.write("," + str(0) + "\n")

            count_3 = 0
            for j in judge[:3]:
                if (j in eva_cft.keys()) and int(eva_cft[j]) <= p_3_count:
                    count_3 += 1
            with open("../wilcoxon_result/p_3_wix_cft_tfidf.txt", "a") as f3b:
                f3b.write(str(count_3))
            count_3 = 0
            for j in judge[:3]:
                if (j in eva_lda.keys()) and int(eva_lda[j]) < p_3_count:
                    count_3 += 1
            with open("../wilcoxon_result/p_3_wix_cft_tfidf.txt", "a") as f3t:
                f3t.write(","+str(count_3)+"\n")

            count_5 = 0
            for j in judge[:5]:
                if (j in eva_cft.keys()) and int(eva_cft[j]) <= p_5_count:
                    count_5 += 1
            with open("../wilcoxon_result/p_5_wix_cft_tfidf.txt", "a") as f5b:
                f5b.write(str(count_5))
            count_5 = 0
            for j in judge[:5]:
                if (j in eva_lda.keys()) and int(eva_lda[j]) < p_5_count:
                    count_5 += 1
            with open("../wilcoxon_result/p_5_wix_cft_tfidf.txt", "a") as f5t:
                f5t.write("," + str(count_5) + "\n")



if __name__ == "__main__":
    evaluate()


