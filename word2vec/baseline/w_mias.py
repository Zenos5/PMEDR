import os
from collections import OrderedDict
def evaluate():
    count = 0
    for f in os.listdir('../RESULT_lda_15_2_6/combined/'):
        count+=1
        content_list_mias = []
        if os.path.exists('../mias_result/'+f):
            with open('../mias_result/'+f, 'r') as rf:
                content_list_mias =rf.readlines()
        content_list_lda = []
        isExist = os.path.exists('../RESULT/combined/' + f)
        if isExist:
            with open('../RESULT/combined/' + f, 'r') as rf:
                content_list_lda = rf.readlines()
        eva_mias = {}
        number = 0
        for rank, line in enumerate(content_list_mias):
            fileName = line.split('/')[-1].split('.')[0]
            if fileName not in eva_mias.keys():
                eva_mias[fileName] = number
                number+=1
        eva_lda = {}
        for rank, line in enumerate(content_list_lda):
            fileName = line.split(',')[0].replace('\'', '').replace('(', '')
            eva_lda[fileName] = rank

        judge = []
        origin_file_count = 0
        for dataset_file in os.listdir("../../data/MSE_dataset_full/dataset_full/text/" + f.split('.')[0] + "/answers"):
            cleaned_name = dataset_file.split('.')[0]
            origin_file_count+=1
            if True:
                judge.append(cleaned_name)
        judge.sort(key=lambda x: int(x.split("_")[1]), reverse=True)

        # p_3_count = 0
        # p_5_count = 0
        #
        # if origin_file_count >= 3:
        #     p_3_count = 3
        # else:
        #     p_3_count += origin_file_count
        # if origin_file_count >= 5:
        #     p_5_count = 5
        # else:
        #     p_5_count += origin_file_count

        with open("../wilcoxon_result/p_1_wix_mias_base_1.txt", "a") as f1b:
            if (judge[0] in eva_mias.keys()) and int(eva_mias[judge[0]]) == 0:
                f1b.write(str(1))
            else:
                f1b.write(str(0))

        with open("../wilcoxon_result/p_1_wix_mias_base_1.txt", "a") as f1t:
            if (judge[0] in eva_lda.keys()) and int(eva_lda[judge[0]]) == 0:
                f1t.write("," + str(1) + "\n")
            else:
                f1t.write("," + str(0) + "\n")

        # count_3 = 0
        # for j in judge[:3]:
        #     if (j in eva_mias.keys()) and int(eva_mias[j]) < p_3_count:
        #         count_3 += 1
        # with open("../wilcoxon_result/p_3_wix_mias_tfidf.txt", "a") as f3b:
        #     f3b.write(str(count_3))
        # count_3 = 0
        # for j in judge[:3]:
        #     if (j in eva_lda.keys()) and int(eva_lda[j]) < p_3_count:
        #         count_3 += 1
        # with open("../wilcoxon_result/p_3_wix_mias_tfidf.txt", "a") as f3t:
        #     f3t.write("," + str(count_3) + "\n")
        #
        # count_5 = 0
        # for j in judge[:5]:
        #     if (j in eva_mias.keys()) and int(eva_mias[j]) < p_5_count:
        #         count_5 += 1
        # with open("../wilcoxon_result/p_5_wix_mias_tfidf.txt", "a") as f5b:
        #     f5b.write(str(count_5))
        # count_5 = 0
        # for j in judge[:5]:
        #     if (j in eva_lda.keys()) and int(eva_lda[j]) < p_5_count:
        #         count_5 += 1
        # with open("../wilcoxon_result/p_5_wix_mias_tfidf.txt", "a") as f5t:
        #     f5t.write("," + str(count_5) + "\n")


if __name__ == "__main__":
    evaluate()