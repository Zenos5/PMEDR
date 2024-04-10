import os
import gensim
import collections
import random
import glob
import pathlib
import argparse
import numpy as np
import re
import math
from bs4 import BeautifulSoup


# Training Corpus: Answers
# Testing Corpus: Questions 
def read_form2vec_corpus(data_path, qa_dict, testlist, trainlist, tokens_only=False):
    qa_list = glob.glob(os.path.join(data_path, '*'))
    out_list = []
    for qa in qa_list:
        # print("QA:",qa)
        answer_dict = {}
        question_list = []
        qa_type = "answers"
        if tokens_only:
            qa_type = "question"
        file_dir = os.path.join(qa, qa_type)
        file_list = os.listdir(file_dir)
        if len(file_list) <= 0:
            print(file_dir, " is empty")
        for file_name in file_list:
            # print("file_name:",file_name)
            file_path = os.path.join(file_dir, file_name)
            with open(file_path, 'r') as f:
                text = f.read()
                delimiters = "><", "<", ">"  # "\n", "\t" , " "
                regex_pattern = '|'.join(map(re.escape, delimiters))
                s = re.sub(r"\s+", "", text)
                tokens = re.split(regex_pattern, s)
                tokens.pop(0)
                while '' in tokens: tokens.remove('')
                tag = pathlib.Path(qa).stem + "_" + file_name[0:file_name.index(".")]
                if tokens_only:
                    if len(question_list) > 0:
                        question_list += tokens
                    else:
                        question_list = tokens
                        if pathlib.Path(qa).stem not in qa_dict.keys():
                            qa_dict[pathlib.Path(qa).stem] = []
                        testlist.append(pathlib.Path(qa).stem)
                    # yield tokens
                else:
                    # For training data, add tags
                    if tag in answer_dict.keys():
                        answer_dict[tag] += tokens
                    else:
                        answer_dict[tag] = tokens
                        if pathlib.Path(qa).stem not in qa_dict.keys():
                            qa_dict[pathlib.Path(qa).stem] = []
                        qa_dict[pathlib.Path(qa).stem].append(tag)
                        trainlist.append(tag)
                    # yield gensim.models.doc2vec.TaggedDocument(tokens, [tag])
        if tokens_only:
            out_list.append(question_list)
            # yield question_list
        else:
            # traindict[pathlib.Path(qa).stem] = []
            for key in answer_dict.keys():
                # traindict[pathlib.Path(qa).stem].append(key)
                out_list.append(gensim.models.doc2vec.TaggedDocument(answer_dict[key], [key]))
    return out_list


# Training Corpus: Answers
# Testing Corpus: Questions 
def read_doc2vec_corpus(data_path, qa_dict, testlist, trainlist, tokens_only=False):
    qa_list = glob.glob(os.path.join(data_path, '*'))
    for qa in qa_list:
        qa_type = "answers"
        if tokens_only:
            qa_type = "question"
        file_dir = os.path.join(qa, qa_type)
        file_list = os.listdir(file_dir)
        for file_name in file_list:
            file_path = os.path.join(file_dir, file_name)
            with open(file_path, 'r') as f:
                text = f.read()
                soup = BeautifulSoup(text, features="html.parser")
                text = soup.get_text('\n')
                tokens = gensim.utils.simple_preprocess(text)
                if tokens_only:
                    testlist.append(pathlib.Path(qa).stem)
                    if pathlib.Path(qa).stem not in qa_dict.keys():
                        qa_dict[pathlib.Path(qa).stem] = []
                    yield tokens
                else:
                    tag = pathlib.Path(qa).stem + "_" + file_name[0:file_name.index(".")]
                    # For training data, add tags
                    trainlist.append(tag)
                    if pathlib.Path(qa).stem not in qa_dict.keys():
                        qa_dict[pathlib.Path(qa).stem] = []
                    qa_dict[pathlib.Path(qa).stem].append(tag)
                    yield gensim.models.doc2vec.TaggedDocument(tokens, [tag])


def test_question(doc_id, model, qa_dict, testlist, trainlist, test_corpus, train_corpus):
    print(test_corpus[doc_id])
    inferred_vector = model.infer_vector(test_corpus[doc_id])
    sims = model.dv.most_similar([inferred_vector], topn=len(model.dv))

    # Compare and print the most/median/least similar documents from the train corpus
    print('\nTest Document ({}, {}): «{}»\n'.format(doc_id, testlist[doc_id], ' '.join(test_corpus[doc_id])))
    print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % model)
    for label, index in [('MOST', 0), ('MEDIAN', len(sims) // 2), ('LEAST', len(sims) - 1)]:
        print(u'%s %s: «%s»\n' % (label, sims[index], ' '.join(train_corpus[trainlist.index(sims[index][0])].words)))

    print(f"\nTrue Answers to Question {testlist[doc_id]}")

    for answer in qa_dict[testlist[doc_id]]:
        answer_id = trainlist.index(answer)
        index = [docid for docid, sim in sims].index(answer)
        print(index, sims[index])
        print(u'%s: «%s»\n' % (sims[index], ' '.join(train_corpus[answer_id].words)))


def answer_confidence(doc_id, model, qa_dict, testlist, trainlist, test_corpus):
    inferred_vector = model.infer_vector(test_corpus[doc_id])
    sims = model.dv.most_similar([inferred_vector], topn=len(model.dv))

    conf_dict = {}

    for answer in qa_dict[testlist[doc_id]]:
        answer_id = trainlist.index(answer)
        index = [docid for docid, sim in sims].index(answer)
        print(index, sims[index], "=", sims[index][1])
        conf_dict[sims[index][0]] = sims[index][1]

    return conf_dict, sims

def calc_metrics(sims_dict, ideal_dict):
    count = 0
    rel_pos = 0
    p_1 = 0
    p_3 = 0
    p_5 = 0
    p_10 = 0
    dcg = []
    idcg = []
    ndcg = []
    for key in sims_dict.keys()[:10]:
        count += 1
        if key in ideal_dict.keys():
            if rel_pos == 0:
                rel_pos = count
            if count <= 1:
                p_1 += 1
                dcg.append[ideal_dict[key] * 1.0]
            if count <= 3:
                p_3 += 1
            if count <= 5:
                p_5 += 1
            if count <= 10:
                p_10 += 1
            if count > 1:
                dcg.append(dcg[-1] + ideal_dict[key] / math.log2(count))
        else:
            if count <= 1:
                dcg.append(0.0)
            else:
                dcg.append(dcg[-1])
    for i in range(10):
        if len(ideal_dict.keys()) < i:
            key = ideal_dict.keys()[i]
            if i <= 0:
                idcg.append(ideal_dict[key] * 1.0)
            else:
                idcg.append(idcg[-1] + ideal_dict[key] / math.log2(i + 1))
        else:
            idcg.append(idcg[-1])
    rel_num = len(ideal_dict.keys())
    rr = 0.0
    if rel_pos > 0:
        rr = 1.0 / rel_pos
    precision = [p_1, p_3 / 3.0, p_5 / 5.0, p_10 / 10.0]
    recall = [p_1 / rel_num, p_3 / rel_num, p_5 / rel_num, p_10 / rel_num]
    
    ndcg = [0.0 if idcg[0] == 0.0 else dcg[0] / idcg[0], 
            0.0 if idcg[2] == 0.0 else dcg[2] / idcg[2], 
            0.0 if idcg[4] == 0.0 else dcg[4] / idcg[4], 
            0.0 if idcg[9] == 0.0 else dcg[9] / idcg[9]]
    p_dcg = [dcg[0], dcg[2], dcg[4], dcg[9]]
    i_dcg = [idcg[0], idcg[2], idcg[4], idcg[9]]

    return precision, recall, rr, p_dcg, i_dcg, ndcg

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="running doc2vec combined with formula2vec")
    parser.add_argument("--test", type=bool, default=False)
    parser.add_argument("--eval", type=bool, default=False)
    parser.add_argument("--compare", type=bool, default=False)
    parser.add_argument("--metrics", type=bool, default=False)
    parser.add_argument("--text-data-dir", type=str, default="../data/MSE_dataset_full/dataset_full/text/")
    parser.add_argument("--math-data-dir", type=str, default="../data/MSE_dataset_full/dataset_full/math/")
    parser.add_argument("--f2v-checkpoint", type=str, default="doc2vec/checkpoints/f2v_CBOW/f2v_50.model")
    parser.add_argument("--d2v-checkpoint", type=str, default="doc2vec/checkpoints/d2v_CBOW/d2v_50.model")
    parser.add_argument("--question", type=str, default="")
    args = parser.parse_args()
    print("Processing")

    # Set file names for train and test data
    text_data_dir = args.text_data_dir
    math_data_dir = args.math_data_dir
    d2v_checkpoint = args.d2v_checkpoint
    f2v_checkpoint = args.f2v_checkpoint

    d2v_trainlist = []
    d2v_testlist = []
    d2v_qa_dict = {}
    f2v_trainlist = []
    f2v_testlist = []
    f2v_qa_dict = {}

    print("test corpus")
    d2v_test_corpus = list(read_doc2vec_corpus(text_data_dir, d2v_qa_dict, tokens_only=True, testlist=d2v_testlist,
                                               trainlist=d2v_trainlist))
    f2v_test_corpus = list(read_form2vec_corpus(math_data_dir, f2v_qa_dict, tokens_only=True, testlist=f2v_testlist,
                                                trainlist=f2v_trainlist))
    print("train corpus")
    d2v_train_corpus = list(
        read_doc2vec_corpus(text_data_dir, d2v_qa_dict, testlist=d2v_testlist, trainlist=d2v_trainlist))
    f2v_train_corpus = list(
        read_form2vec_corpus(math_data_dir, f2v_qa_dict, testlist=f2v_testlist, trainlist=f2v_trainlist))

    print("d2v data:", d2v_train_corpus[:2])
    print(d2v_test_corpus[:2])

    print(d2v_trainlist[0:2])
    print(d2v_testlist[0:2], "\n")

    print("f2v data:", f2v_train_corpus[:2])
    print(f2v_test_corpus[:2])

    print(f2v_trainlist[0:2])
    print(f2v_testlist[0:2], "\n")

    # Training the Model
    d2v_model = gensim.models.doc2vec.Doc2Vec.load(d2v_checkpoint)
    f2v_model = gensim.models.doc2vec.Doc2Vec.load(f2v_checkpoint)

    print("Finished setting up the model")

    if args.test:
        print("\nTEST")
        print("Doc2Vec:")
        # Pick a random document from the test corpus and infer a vector from the model
        doc_id = random.randint(0, len(d2v_test_corpus) - 1)

        inferred_vector = d2v_model.infer_vector(d2v_test_corpus[doc_id])
        sims = d2v_model.dv.most_similar([inferred_vector], topn=len(d2v_model.dv))

        # Compare and print the most/median/least similar documents from the train corpus
        print('Test Document ({}): «{}»\n'.format(doc_id, ' '.join(d2v_test_corpus[doc_id])))
        print(d2v_testlist[doc_id])
        print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % d2v_model)
        for label, index in [('MOST', 0), ('MEDIAN', len(sims) // 2), ('LEAST', len(sims) - 1)]:
            print(u'%s %s: «%s»\n' % (
            label, sims[index], ' '.join(d2v_train_corpus[d2v_trainlist.index(sims[index][0])].words)))

        # Pick 8_0.xml from the test corpus and infer a vector from the model
        doc_id = d2v_testlist.index("8")
        inferred_vector = d2v_model.infer_vector(d2v_test_corpus[doc_id])
        sims = d2v_model.dv.most_similar([inferred_vector], topn=len(d2v_model.dv))

        # Compare and print the most/median/least similar documents from the train corpus
        print('\nTest Document ({}): «{}»\n'.format(doc_id, ' '.join(d2v_test_corpus[doc_id])))
        print(d2v_testlist[doc_id])
        print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % d2v_model)
        for label, index in [('MOST', 0), ('MEDIAN', len(sims) // 2), ('LEAST', len(sims) - 1)]:
            print(u'%s %s: «%s»\n' % (
            label, sims[index], ' '.join(d2v_train_corpus[d2v_trainlist.index(sims[index][0])].words)))

        print(f"\nTrue Answers to Question {d2v_testlist[doc_id]}")

        # /home/aw742/Word2vec/data/MSE_dataset_full/dataset_full/text/8/answers/17_28.0.txt
        answer_id = d2v_trainlist.index("8_17_28")
        index = [docid for docid, sim in sims].index(d2v_trainlist[answer_id])
        print(index)
        print(u'%s: «%s»\n' % (sims[index], ' '.join(d2v_train_corpus[answer_id].words)))
        # /home/aw742/Word2vec/data/MSE_dataset_full/dataset_full/text/8/answers/24_5.0.txt
        answer_id = d2v_trainlist.index("8_24_5")
        index = [docid for docid, sim in sims].index(d2v_trainlist[answer_id])
        print(index)
        print(u'%s: «%s»\n' % (sims[index], ' '.join(d2v_train_corpus[answer_id].words)))
        # /home/aw742/Word2vec/data/MSE_dataset_full/dataset_full/text/8/answers/26_4.0.txt
        answer_id = d2v_trainlist.index("8_26_4")
        index = [docid for docid, sim in sims].index(d2v_trainlist[answer_id])
        print(index)
        print(u'%s: «%s»\n' % (sims[index], ' '.join(d2v_train_corpus[answer_id].words)))
        # /home/aw742/Word2vec/data/MSE_dataset_full/dataset_full/text/8/answers/319_34.0.txt
        answer_id = d2v_trainlist.index("8_319_34")
        index = [docid for docid, sim in sims].index(d2v_trainlist[answer_id])
        print(index)
        print(u'%s: «%s»\n' % (sims[index], ' '.join(d2v_train_corpus[answer_id].words)))
        # /home/aw742/Word2vec/data/MSE_dataset_full/dataset_full/text/8/answers/931_32.0.txt
        answer_id = d2v_trainlist.index("8_931_32")
        index = [docid for docid, sim in sims].index(d2v_trainlist[answer_id])
        print(index)
        print(u'%s: «%s»\n' % (sims[index], ' '.join(d2v_train_corpus[answer_id].words)))
        # /home/aw742/Word2vec/data/MSE_dataset_full/dataset_full/text/8/answers/934_4.0.txt
        answer_id = d2v_trainlist.index("8_934_4")
        index = [docid for docid, sim in sims].index(d2v_trainlist[answer_id])
        print(index)
        print(u'%s: «%s»\n' % (sims[index], ' '.join(d2v_train_corpus[answer_id].words)))
        # /home/aw742/Word2vec/data/MSE_dataset_full/dataset_full/text/8/answers/396100_3.0.txt
        answer_id = d2v_trainlist.index("8_396100_3")
        index = [docid for docid, sim in sims].index(d2v_trainlist[answer_id])
        print(index)
        print(u'%s: «%s»\n' % (sims[index], ' '.join(d2v_train_corpus[answer_id].words)))
        # /home/aw742/Word2vec/data/MSE_dataset_full/dataset_full/text/8/answers/3693845_3.0.txt
        answer_id = d2v_trainlist.index("8_3693845_3")
        index = [docid for docid, sim in sims].index(d2v_trainlist[answer_id])
        print(index)
        print(u'%s: «%s»\n' % (sims[index], ' '.join(d2v_train_corpus[answer_id].words)))

        print("\nFormula2Vec:")

        # Pick a random document from the test corpus and infer a vector from the model
        doc_id = f2v_testlist.index(d2v_testlist[doc_id])
        inferred_vector = f2v_model.infer_vector(f2v_test_corpus[doc_id])
        sims = f2v_model.dv.most_similar([inferred_vector], topn=len(f2v_model.dv))

        # Compare and print the most/median/least similar documents from the train corpus
        print('Test Document ({}): «{}»\n'.format(doc_id, ' '.join(f2v_test_corpus[doc_id])))
        print(f2v_testlist[doc_id])
        print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % f2v_model)
        for label, index in [('MOST', 0), ('MEDIAN', len(sims) // 2), ('LEAST', len(sims) - 1)]:
            print(u'%s %s: «%s»\n' % (
            label, sims[index], ' '.join(f2v_train_corpus[f2v_trainlist.index(sims[index][0])].words)))

        # Pick 8_0.xml from the test corpus and infer a vector from the model
        doc_id = f2v_testlist.index("8")
        inferred_vector = f2v_model.infer_vector(f2v_test_corpus[doc_id])
        sims = f2v_model.dv.most_similar([inferred_vector], topn=len(f2v_model.dv))

        # Compare and print the most/median/least similar documents from the train corpus
        print('\nTest Document ({}): «{}»\n'.format(doc_id, ' '.join(f2v_test_corpus[doc_id])))
        print(f2v_testlist[doc_id])
        print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % f2v_model)
        for label, index in [('MOST', 0), ('MEDIAN', len(sims) // 2), ('LEAST', len(sims) - 1)]:
            print(u'%s %s: «%s»\n' % (
            label, sims[index], ' '.join(f2v_train_corpus[f2v_trainlist.index(sims[index][0])].words)))

        print(f"\nTrue Answers to Question {f2v_testlist[doc_id]}")

        # /home/aw742/Word2vec/data/MSE_dataset_full/dataset_full/math/8/answers/319_34.0_0.xml
        answer_id = f2v_trainlist.index("8_319_34")
        index = [docid for docid, sim in sims].index(f2v_trainlist[answer_id])
        print(index)
        print(u'%s: «%s»\n' % (sims[index], ' '.join(f2v_train_corpus[answer_id].words)))
        # /home/aw742/Word2vec/data/MSE_dataset_full/dataset_full/math/8/answers/931_32.0_0.xml
        answer_id = f2v_trainlist.index("8_931_32")
        index = [docid for docid, sim in sims].index(f2v_trainlist[answer_id])
        print(index)
        print(u'%s: «%s»\n' % (sims[index], ' '.join(f2v_train_corpus[answer_id].words)))
        # /home/aw742/Word2vec/data/MSE_dataset_full/dataset_full/math/8/answers/396100_3.0_0.xml
        answer_id = f2v_trainlist.index("8_396100_3")
        index = [docid for docid, sim in sims].index(f2v_trainlist[answer_id])
        print(index)
        print(u'%s: «%s»\n' % (sims[index], ' '.join(f2v_train_corpus[answer_id].words)))
        # /home/aw742/Word2vec/data/MSE_dataset_full/dataset_full/math/8/answers/3693845_3.0_0.xml
        answer_id = f2v_trainlist.index("8_3693845_3")
        index = [docid for docid, sim in sims].index(f2v_trainlist[answer_id])
        print(index)
        print(u'%s: «%s»\n' % (sims[index], ' '.join(f2v_train_corpus[answer_id].words)))

        print("Tested the model")

    if args.eval:
        print("\nEVAL")

        print("\nDoc2Vec:")

        if len(args.question) <= 0:
            d2v_rand_question = random.randrange(len(d2v_testlist))
            while d2v_testlist[d2v_rand_question] not in f2v_testlist:
                d2v_rand_question = random.randrange(len(d2v_testlist))
        else:
            d2v_rand_question = d2v_testlist.index(args.question)
        f2v_rand_question = f2v_testlist.index(args.question)

        test_question(d2v_rand_question, d2v_model, d2v_qa_dict, d2v_testlist, d2v_trainlist, d2v_test_corpus,
                      d2v_train_corpus)

        print("\nFormula2Vec:")
        
        test_question(f2v_rand_question, f2v_model, f2v_qa_dict, f2v_testlist, f2v_trainlist, f2v_test_corpus,
                      f2v_train_corpus)

    if args.compare:
        print("\nCOMPARE")
        overall_conf = {}

        if len(args.question) <= 0:
            d2v_rand_question = random.randrange(len(d2v_testlist))
            while d2v_testlist[d2v_rand_question] not in f2v_testlist:
                d2v_rand_question = random.randrange(len(d2v_testlist))
        else:
            d2v_rand_question = d2v_testlist.index(args.question)
        f2v_rand_question = f2v_testlist.index(args.question)

        print(f"\nTrue Answers to Question {d2v_testlist[d2v_rand_question]}")
        print("\nDoc2Vec:")
        d2v_conf, _ = answer_confidence(d2v_rand_question, d2v_model, d2v_qa_dict, d2v_testlist, d2v_trainlist,
                                     d2v_test_corpus)
        print("\nFormula2Vec:")
        f2v_conf, _ = answer_confidence(f2v_rand_question, f2v_model, f2v_qa_dict, f2v_testlist, f2v_trainlist,
                                     f2v_test_corpus)

        print()
        print(d2v_conf.keys())
        print(f2v_conf.keys())

        for key in d2v_conf.keys():
            overall_conf[key] = 0.5 * d2v_conf[key]

        for key in f2v_conf.keys():
            if key in overall_conf.keys():
                overall_conf[key] += 0.5 * f2v_conf[key]
            else:
                overall_conf[key] = 0.5 * f2v_conf[key]

        for key in overall_conf.keys():
            print(key)
            print("\t", overall_conf[key])
    
    if args.metrics:
        print("\nMETRICS")
        if len(args.question) <= 0:
            q_set = set()
            q_set.update(d2v_testlist)
            q_set.update(f2v_testlist)
            a_prec = []
            a_rec = []
            mrr = []
            a_dcg = []
            a_idcg = []
            a_ndcg = []
            for q in q_set:
                a_dict = {}
                if q in d2v_testlist:
                    dq_id = d2v_testlist.index(q)
                    inferred_vector = d2v_model.infer_vector(d2v_test_corpus[dq_id])
                    d_sims = d2v_model.dv.most_similar([inferred_vector], topn=len(d2v_model.dv))
                    for da, dsim in d_sims:
                        a_dict[da] = 0.5 * dsim
                if q in f2v_testlist:
                    fq_id = f2v_testlist.index(q)
                    inferred_vector = f2v_model.infer_vector(f2v_test_corpus[fq_id])
                    f_sims = f2v_model.dv.most_similar([inferred_vector], topn=len(f2v_model.dv))
                    for fa, fsim in f_sims:
                        if fa in a_dict.keys():
                            a_dict[fa] += 0.5 * fsim
                        else:
                            a_dict[fa] = 0.5 * fsim
                if q not in f2v_testlist and q not in d2v_testlist:
                    a_dict[da] = 0.0
                a_dict = sorted(a_dict.items(), key=lambda x: x[1], reverse=True)
                ideal_dict = {}
                for answer in d2v_qa_dict[q]:
                    rank = int(answer[answer.rfind("_"):])
                    ideal_dict[answer] = rank
                for answer in f2v_qa_dict[q]:
                    if answer not in ideal_dict.keys():
                        rank = int(answer[answer.rfind("_"):])
                        ideal_dict[answer] = rank
                ideal_dict = sorted(ideal_dict.items(), key=lambda x: x[1], reverse=True)
                precision, recall, rr, p_dcg, i_dcg, n_dcg = calc_metrics(a_dict, ideal_dict)
                a_prec.append(precision)
                a_rec.append(recall)
                mrr.append(rr)
                a_dcg.append(p_dcg)
                a_idcg.append(i_dcg)
                a_ndcg.append(n_dcg)
            print("\nP@1 P@3 P@5 P@10")
            print(np.mean(precision, axis=1))
            print("\nR@1 R@3 R@5 R@10")
            print(np.mean(recall, axis=1))
            print("\nRR:", np.mean(rr))
            print("\nDCG@1 DCG@3 DCG@5 DCG@10")
            print(np.mean(p_dcg, axis=1))
            print("\niDCG@1 iDCG@3 iDCG@5 iDCG@10")
            print(np.mean(i_dcg, axis=1))
            print("\nnDCG@1 nDCG@3 nDCG@5 nDCG@10")
            print(np.mean(n_dcg, axis=1))
        else:
            a_dict = {}
            if args.question in d2v_testlist:
                dq_id = d2v_testlist.index(args.question)
                inferred_vector = d2v_model.infer_vector(d2v_test_corpus[dq_id])
                d_sims = d2v_model.dv.most_similar([inferred_vector], topn=len(d2v_model.dv))
                for da, dsim in d_sims:
                    a_dict[da] = 0.5 * dsim
            if args.question in f2v_testlist:
                fq_id = f2v_testlist.index(args.question)
                inferred_vector = f2v_model.infer_vector(f2v_test_corpus[fq_id])
                f_sims = f2v_model.dv.most_similar([inferred_vector], topn=len(f2v_model.dv))
                for fa, fsim in f_sims:
                    if fa in a_dict.keys():
                        a_dict[fa] += 0.5 * fsim
                    else:
                        a_dict[fa] = 0.5 * fsim
            if args.question not in f2v_testlist and args.question not in d2v_testlist:
                a_dict[da] = 0.0
            a_dict = sorted(a_dict.items(), key=lambda x: x[1], reverse=True)
            ideal_dict = {}
            for answer in d2v_qa_dict[args.question]:
                rank = int(answer[answer.rfind("_"):])
                ideal_dict[answer] = rank
            for answer in f2v_qa_dict[args.question]:
                if answer not in ideal_dict.keys():
                    rank = int(answer[answer.rfind("_"):])
                    ideal_dict[answer] = rank
            ideal_dict = sorted(ideal_dict.items(), key=lambda x: x[1], reverse=True)
            precision, recall, rr, p_dcg, i_dcg, n_dcg = calc_metrics(a_dict, ideal_dict)

            print("\nP@1 P@3 P@5 P@10")
            print(precision)
            print("\nR@1 R@3 R@5 R@10")
            print(recall)
            print("\nRR:", rr)
            print("\nDCG@1 DCG@3 DCG@5 DCG@10")
            print(p_dcg)
            print("\niDCG@1 iDCG@3 iDCG@5 iDCG@10")
            print(i_dcg)
            print("\nnDCG@1 nDCG@3 nDCG@5 nDCG@10")
            print(n_dcg)
