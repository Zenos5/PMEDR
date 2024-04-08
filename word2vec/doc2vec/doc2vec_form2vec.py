import os
import gensim
import collections
import random
import glob
import pathlib
import argparse
import numpy as np
import re
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
        index = [docid for docid, sim in sims].index(trainlist[answer_id])
        print(index)
        print(u'%s: «%s»\n' % (sims[index], ' '.join(train_corpus[answer_id].words)))


def answer_confidence(doc_id, model, qa_dict, testlist, trainlist, test_corpus):
    inferred_vector = model.infer_vector(test_corpus[doc_id])
    sims = model.dv.most_similar([inferred_vector], topn=len(model.dv))

    conf_dict = {}

    for answer in qa_dict[testlist[doc_id]]:
        answer_id = trainlist.index(answer)
        index = [docid for docid, sim in sims].index(trainlist[answer_id])
        conf_dict[sims[index][0]] = sims[index][1]

    return conf_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="running doc2vec combined with formula2vec")
    parser.add_argument("--test", type=bool, default=False)
    parser.add_argument("--eval", type=bool, default=False)
    parser.add_argument("--compare", type=bool, default=False)
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
        else:
            d2v_rand_question = d2v_testlist.index(args.question)
        while d2v_testlist[d2v_rand_question] not in f2v_testlist:
            d2v_rand_question = random.randrange(len(d2v_testlist))
        f2v_rand_question = f2v_testlist.index(d2v_testlist[d2v_rand_question])

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
        else:
            d2v_rand_question = d2v_testlist.index(args.question)
        while d2v_testlist[d2v_rand_question] not in f2v_testlist:
            d2v_rand_question = random.randrange(len(d2v_testlist))
        f2v_rand_question = f2v_testlist.index(d2v_testlist[d2v_rand_question])

        print(f"\nTrue Answers to Question {d2v_testlist[d2v_rand_question]}")
        d2v_conf = answer_confidence(d2v_rand_question, d2v_model, d2v_qa_dict, d2v_testlist, d2v_trainlist,
                                     d2v_test_corpus)

        f2v_conf = answer_confidence(f2v_rand_question, f2v_model, f2v_qa_dict, f2v_testlist, f2v_trainlist,
                                     f2v_test_corpus)

        for key in d2v_conf.keys():
            overall_conf[key] = 0.5 * d2v_conf[key]

        for key in f2v_conf.keys():
            if key in overall_conf.keys():
                overall_conf[key] += 0.5 * d2v_conf[key]
            else:
                overall_conf[key] = 0.5 * d2v_conf[key]

        for key in overall_conf.keys():
            print(key)
            print("\t", overall_conf[key])