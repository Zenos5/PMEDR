# Run this as a main function
import argparse

import numpy as np

import treeMatch_patical_match2 as tm
import os
import KL
import text_clean as tc
import time
def math_match(qm, am, question_id, path):
    newTree = tm.FastTreeMatch()
    final_result = []
    for target in qm:
        math_result = {}
        math_result = newTree.run(target, am)
        final_result.append(math_result)
    temp = {}
    if final_result:
        for d in final_result:
            if d:
                for k in d.keys():
                    temp[k] = temp.get(k, 0) + d[k]
        for k in temp.keys():
            temp[k] = temp[k] / len(final_result)
        result = {}
        for k in temp.keys():
            new_key = ('_').join(k.split('/')[-1].split('_')[:2])[:-1][:-1]

            if new_key not in result:
                result[new_key] = [temp[k]]
            else:
                result[new_key].append(temp[k])
        for k in result.keys():
            result[k]= sum(result[k])/len(result[k])
        if result:
            result = normalization(result)
            result = dict(sorted(result.items(), key=lambda x: x[1], reverse=True))
            result_path = path+'/math/'
            isExist = os.path.exists(result_path)
            if not isExist:
                # Create a new directory because it does not exist
                os.makedirs(result_path)
            file = result_path + question_id + '.txt'
            with open(file, 'w') as f:
                for r in result.keys():
                    f.write(f"{r, result[r]}\n")

        return result
    else:
        return None

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

def use_normal_kl(cl,question_words, question_dic, atf):
    kl = KL.kl_divergence()
    text_result = {}
    for afn in atf:
        with open(afn, 'r') as af:
             pa= af.read()

        answer_words = cl.remove_stop(pa)
        answer_words = [x for x in answer_words if not (x.isdigit()
                                                        or x[0] == '-' and x[1:].isdigit())]
        answer_dic = cl.indexing(cl.stem(answer_words))
        # print("answer")
        # print(answer_dic)
        key = ('_').join(afn.split('/')[-1].split('_')[:2]).split('.')[0]
        # print("len a: ", len(answer_words))
        text_result[key] = kl.kl_archive_divergence(question_dic, len(question_words), answer_dic, len(answer_words))
    return text_result

def use_new_tfidf(cl,question_words, question_dic, atf):
    kl = KL.kl_divergence()
    text_result = {}
    answer_list=[]
    answer_word_list = []
    for afn in atf:
        with open(afn, 'r') as af:
             pa= af.read()
        answer_words = cl.remove_stop(pa)
        answer_words=[x for x in answer_words if not (x.isdigit()
                                           or x[0] == '-' and x[1:].isdigit())]
        answer_dic = cl.indexing(cl.stem(answer_words))
        answer_word_list.append(answer_words)
        answer_list.append(answer_dic)
    for answerDict, afn,  answer_words in zip(answer_list, atf, answer_word_list):
        key = ('_').join(afn.split('/')[-1].split('_')[:2]).split('.')[0]
        result= kl.new_kl_tfidf(question_dic, len(question_words), answer_list, answerDict, len(answer_words))
        text_result[key] = result
    return text_result
def use_kl_tfidf(cl,question_words, question_dic, atf):
    kl = KL.kl_divergence()
    text_result = {}
    answer_list=[]
    for afn in atf:
        with open(afn, 'r') as af:
             pa= af.read()
        answer_words = cl.remove_stop(pa)
        answer_words=[x for x in answer_words if not (x.isdigit()
                                           or x[0] == '-' and x[1:].isdigit())]
        answer_dic = cl.indexing(cl.stem(answer_words))
        answer_list.append(answer_dic)
    for answerDict, afn in zip(answer_list, atf):
        key = ('_').join(afn.split('/')[-1].split('_')[:2]).split('.')[0]
        result= kl.KL_TF_IDF(question_dic, len(question_words), answer_list, answerDict)
        text_result[key] = result
    return text_result



def use_lda(cl,question_words, question_dic, atf):
    kl = KL.kl_divergence()
    text_result = {}
    if len(question_dic) !=0:
        topicList, keywordMap, numTopics, numKeywords = kl.getLDA_Topics(question_dic)
        for afn in atf:
            with open(afn, 'r') as af:
                pa = af.read()
            answer_words = cl.remove_stop(pa)
            answer_words = [x for x in answer_words if not (x.isdigit()
                                                            or x[0] == '-' and x[1:].isdigit())]
            answer_dic = cl.indexing(cl.stem(answer_words))
            key = ('_').join(afn.split('/')[-1].split('_')[:2]).split('.')[0]
            text_result[key] = kl.kl_lda(question_dic, len(question_words), answer_dic, len(answer_words),keywordMap,numTopics, numKeywords)
    return text_result

def text_match(qtf, atf, question_id, path, type):
    #Text Match
    cl = tc.clean()
    with open(qtf, 'r') as qf:
        qt= qf.read()
    question_words = cl.remove_stop(qt)
    question_words = [x for x in question_words if not (x.isdigit()
                                         or x[0] == '-' and x[1:].isdigit())]
    question_dic = cl.indexing(cl.stem(question_words))
    # print("len q: ", len(question_words))
    # print("question")
    # print(qtf)
    # print(question_dic)
    if type=="tfidf":
        print('tfidf')
        text_result= use_kl_tfidf(cl, question_words, question_dic, atf)
    elif type=="tfidf_new":
        print('tfidf_new')
        text_result= use_new_tfidf(cl, question_words, question_dic, atf)

    elif type=="lda":
        print('lda')
        text_result= use_lda(cl, question_words, question_dic, atf)
    elif type=="base":
        print('base')
        text_result= use_normal_kl(cl, question_words, question_dic, atf)
    else:
        print('use normal')
        text_result = use_normal_kl(cl, question_words, question_dic, atf)

    if text_result:
        for key in text_result:
            # text_result[key] = np.absolute(text_result[key])
            text_result[key] = (-1)*text_result[key]
        # print(text_result)
        text_result = normalization(text_result)
        # print(text_result)
        text_result = dict(sorted(text_result.items(), key=lambda x: x[1], reverse=True))

    # result_path = path+'/text/'
    # isExist = os.path.exists(result_path)
    # if not isExist:
    #     # Create a new directory because it does not exist
    #     os.makedirs(result_path)
    # file = result_path + question_id + '.txt'
    # with open(file, 'w') as f:
    #     for r in text_result.keys():
    #         f.write(f"{r, text_result[r]}\n")

    return text_result


if __name__ == "__main__":
    st = time.process_time()
    path = 'lda_15_3_4'
    type='lda'
    # print(path)
    parser = argparse.ArgumentParser()
    parser.add_argument('--mathdata', type=str, help="question formulas",default="../../data/MSE_dataset_full/dataset_full/math")
    parser.add_argument('--textdata', type=str, help="question text",default="../../data/MSE_dataset_full/dataset_full/text")

    args = vars(parser.parse_args())
    math_folder = args['mathdata']
    text_folder = args['textdata']
    testcase = 0
    for dirt in os.listdir(math_folder):
        if  dirt != ".DS_Store":
            qm = []
            qmd= math_folder+'/'+dirt+'/question'
            for f in os.listdir(qmd):
                if f != ".DS_Store":
                    qm.append(qmd + '/' + f)
            am=[]
            amd = math_folder + '/' + dirt + '/answers'
            if not os.path.exists(amd):
                # print("d", dirt)
                continue
            else:
                for f in os.listdir(amd):
                    if f != ".DS_Store":
                        am.append(amd + '/' + f)
                qtf=''
                qtd = text_folder+'/'+dirt+'/question'
                for f in os.listdir(qtd):
                    if f != ".DS_Store":
                        qtf=qtd + '/' + f
                atf=[]
                atd = text_folder + '/' + dirt + '/answers'
                for f in os.listdir(atd):
                    if f != ".DS_Store":
                        atf.append(atd + '/' + f)
                math_result = math_match(qm, am, dirt, path)
                text_result = text_match(qtf, atf, dirt, path, type)

                final_result = {}
                if text_result:
                    for key in text_result.keys():
                        if math_result and (key in math_result):
                            final_result[key] = (math_result[key]+text_result[key])/2
                        # else:
                        #     final_result[key]= text_result[key]
                if final_result:
                    final_result = sorted(final_result.items(), key=lambda x: x[1], reverse=True)

                    result_path = path+'/combined/'
                    isExist = os.path.exists(result_path)
                    if not isExist:
                        # Create a new directory because it does not exist
                        os.makedirs(result_path)
                    file = result_path + dirt + '.txt'
                    with open(file, 'w') as f:
                        for r in final_result:
                            f.write(f"{r}\n")
        testcase +=1
    et = time.process_time()
    # print("execute time: ", et-st)
# with open("w_result.txt", 'a') as w_f:
                #     for k in final_result1.keys():
                #         w_f.write(str(final_result1[k])+','+str(final_result2[k])+"\n")
                #         print(str(final_result1[k])+','+str(final_result2[k])+"\n")

