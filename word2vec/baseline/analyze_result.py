# Run this as a main function
import argparse
import numpy as np

import treeMatch_patical_match2 as tm
import os
import sys
import text_clean as tc
import collections, functools, operator
import math


def normalization(text_result):

    fn = lambda value, x_max, x_min: (value - x_min) / (x_max - x_min)

        # find min and max values
    max_x = max(text_result.values())

    min_x = min(text_result.values())
    if max_x != min_x:
    # normalize each value in the dict
        for k, v in text_result.items():
            text_result[k] = fn(v, max_x, min_x)
    return text_result

def get_common_words(question_dict, answerDict):
    common_keys = set(question_dict).intersection(answerDict)
    return common_keys

def kl_archive_divergence(question_dict, questionWordNum, answerDict, answerWordNum):
    sum = 0
    w_s = []
    common_keys = get_common_words(question_dict, answerDict)
    num_common_q = 0
    for c in common_keys:
        num_common_q+=question_dict[c]
    for word in common_keys:
        a = question_dict[word] / questionWordNum
        b = (answerDict[word] / num_common_q)
        val = (a) * np.log( a / b )    #Sophie version
        sum += val
        w_s.append(val)
    return w_s

def KL_TF_IDF(question_dict, questionWordNum, answer_list, answerDict):
    a_b = []
    w_s = []
    sum = 0
    common_keys = get_common_words(question_dict, answerDict)
    num_common_q=0
    for c in common_keys:
        num_common_q += question_dict[c]
    for word in common_keys:
        a = question_dict[word] / questionWordNum
        tf = answerDict[word]/max(answerDict.values())
        has_word = 0
        for ans in answer_list:
            if word in ans.keys():
                has_word+=1
        if has_word==0:
            has_word=1

        idf = np.log(len(answer_list)/ has_word)
        b = tf*idf
        if b == 0:
            continue
        val = (a) * np.log(a / b)  # Sophie version
        sum += val
        a_b.append(a-b)
        w_s.append(val)
    return a_b, w_s
def use_normal_kl(cl,question_words, question_dic, atf):
    text_result = {}
    score = []
    quest_len = []
    for afn in atf:
        with open(afn, 'r') as af:
             pa= af.read()
        answer_words = cl.remove_stop(pa)
        answer_dic = cl.indexing(cl.stem(answer_words))
        key = ('_').join(afn.split('/')[-1].split('_')[:2]).split('.')[0]
        sum = kl_archive_divergence(question_dic, len(question_words), answer_dic, len(answer_words))
        score.extend(sum)
    return score

def use_kl_tfidf(cl,question_words, question_dic, atf):
    text_result = {}
    answer_list=[]
    for afn in atf:
        with open(afn, 'r') as af:
             pa= af.read()
        answer_words = cl.remove_stop(pa)
        answer_dic = cl.indexing(cl.stem(answer_words))
        answer_list.append(answer_dic)
    score=[]
    quest_len=[]
    for answerDict, afn in zip(answer_list, atf):
        key = ('_').join(afn.split('/')[-1].split('_')[:2]).split('.')[0]

        b_a, sum= KL_TF_IDF(question_dic, len(question_words), answer_list, answerDict)
        score.extend(sum)
        quest_len.extend(b_a)
    return score, quest_len

def text_match(qtf, atf):
    #Text Match
    cl = tc.clean()
    with open(qtf, 'r') as qf:
        qt= qf.read()
    question_words = cl.remove_stop(qt)
    question_dic = cl.indexing(cl.stem(question_words))
    score1= use_normal_kl(cl, question_words, question_dic, atf)
    score2,quest_len= use_kl_tfidf(cl, question_words, question_dic, atf)

    return score1, score2, quest_len



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mathdata', type=str, help="question formulas",default="../../data/MSE_dataset_full/dataset_full/math")
    parser.add_argument('--textdata', type=str, help="question text",default="../../data/MSE_dataset_full/dataset_full/text")

    args = vars(parser.parse_args())
    math_folder = args['mathdata']
    text_folder = args['textdata']
    testcase = 0
    score_norm=[]
    score_tfidf=[]
    question_len=[]
    for dirt in os.listdir(math_folder):
        if testcase >= 10 and testcase < 20:
            if dirt != ".DS_Store":
                qm = []
                qmd = math_folder + '/' + dirt + '/question'
                for f in os.listdir(qmd):
                    if f != ".DS_Store":
                        qm.append(qmd + '/' + f)
                am = []
                amd = math_folder + '/' + dirt + '/answers'
                if not os.path.exists(amd):
                    print("d", dirt)
                    continue
                else:
                    for f in os.listdir(amd):
                        if f != ".DS_Store":
                            am.append(amd + '/' + f)
                    qtf = ''
                    qtd = text_folder + '/' + dirt + '/question'
                    for f in os.listdir(qtd):
                        if f != ".DS_Store":
                            qtf = qtd + '/' + f
                    atf = []
                    atd = text_folder + '/' + dirt + '/answers'
                    for f in os.listdir(atd):
                        if f != ".DS_Store":
                            atf.append(atd + '/' + f)
                    # math_result = math_match(qm, am, dirt)
                    score1, score2, quest_len = text_match(qtf, atf)
                    score_norm.extend(score1)
                    score_tfidf.extend(score2)
                    question_len.extend(quest_len)
                    # break
        if testcase >20:
            break
        testcase +=1
        print(testcase)
    print(len(question_len))
    print("*" * 50)
    print(len(score_tfidf))
    print("*" * 50)
    print(len(score_norm))
    with open("analyze_3.csv", 'w') as f:
        f.write("a-tfidf,base KL,tf_idf KL\n")
        i = 0
        for q, b, t in zip(question_len, score_norm, score_tfidf):
            f.write(str(q)+','+str(b)+','+str(t)+'\n')
            i+=1
            print(i)
