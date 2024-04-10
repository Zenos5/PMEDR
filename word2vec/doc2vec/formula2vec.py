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
def read_corpus(data_path, qa_dict, testlist, trainlist, tokens_only=False):
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
                delimiters = "><", "<", ">" # "\n", "\t" , " "
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
        

def test_question(doc_id, model, qa_dict, testlist, trainlist, train_corpus):
    inferred_vector = model.infer_vector(test_corpus[doc_id])
    sims = model.dv.most_similar([inferred_vector], topn=len(model.dv))

    # Compare and print the most/median/least similar documents from the train corpus
    print('\nTest Document ({}, {}): «{}»\n'.format(doc_id, testlist[doc_id], ' '.join(test_corpus[doc_id])))
    print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % model)
    for label, index in [('MOST', 0), ('MEDIAN', len(sims)//2), ('LEAST', len(sims) - 1)]:
        print(u'%s %s: «%s»\n' % (label, sims[index], ' '.join(train_corpus[trainlist.index(sims[index][0])].words)))
    
    print(f"\nTrue Answers to Question {testlist[doc_id]}")
        
    for answer in qa_dict[testlist[doc_id]]:
        answer_id = trainlist.index(answer)
        index = [docid for docid, sim in sims].index(trainlist[answer_id])
        print(index)
        print(u'%s: «%s»\n' % (sims[index], ' '.join(train_corpus[answer_id].words)))

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
    parser = argparse.ArgumentParser(description="running formula2vec")
    parser.add_argument("--start-epoch", type=int, default=0)
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--train", type=bool, default=False)
    parser.add_argument("--assess", type=bool, default=False)
    parser.add_argument("--test", type=bool, default=False)
    parser.add_argument("--eval", type=bool, default=False)
    parser.add_argument("--metrics", type=bool, default=False)
    parser.add_argument("--question", type=str, default="")
    parser.add_argument("--data-dir", type=str, default="../data/MSE_dataset_full/dataset_full/math/")
    parser.add_argument("--checkpoint", type=str, default="doc2vec/checkpoints/f2v_40.model")
    parser.add_argument("--vector-size", type=int, default=50)
    parser.add_argument("--min-count", type=int, default=1)
    parser.add_argument("--strategy", type=int, default=0) # 0 is DBOW, 1 is DM
    parser.add_argument("--neg-sample", type=int, default=5) # how many “noise words” should be drawn (neg sampling)
    parser.add_argument("--hier-sm", type=int, default=0) # 0 is neg samping, 1 is hier softmax
    args = parser.parse_args()
    print("Processing")

    max_epochs = args.max_epochs
    start_epoch = args.start_epoch

    # Set file names for train and test data
    data_dir = args.data_dir
    checkpoint = args.checkpoint

    trainlist = []
    testlist = []
    qa_dict = {}

    print("test corpus")
    test_corpus = list(read_corpus(data_dir, qa_dict, tokens_only=True, testlist=testlist, trainlist=trainlist))
    print("train corpus")
    train_corpus = list(read_corpus(data_dir, qa_dict, testlist=testlist, trainlist=trainlist))

    print(train_corpus[:2])
    print(test_corpus[:2])

    print(trainlist[0:2])
    print(testlist[0:2])


    # Training the Model
    model = None
    if args.train:
        print("\nTRAIN")
        if start_epoch > 0:
            model = gensim.models.doc2vec.Doc2Vec.load(checkpoint)
        else:
            model = gensim.models.doc2vec.Doc2Vec(vector_size=args.vector_size, min_count=args.min_count, dm=args.strategy, negative=args.neg_sample, hs=args.hier_sm, epochs=max_epochs)
            model.build_vocab(train_corpus)

        print(f"Word '4' appeared {model.wv.get_vecattr('4', 'count')} times in the training corpus.")

        for epoch in range(start_epoch, max_epochs):
            print('iteration {0}'.format(epoch))
            model.train(train_corpus, total_examples=model.corpus_count, epochs=1)
            if epoch % 5 == 0:
                model.save(f"doc2vec/checkpoints/f2v_{epoch + 1}.model")
                print("Model Saved")

        model.save(f"doc2vec/checkpoints/f2v_{max_epochs}.model")
        print("Model Saved")

        vector = model.infer_vector(['<mrow>', '<mn>', '0', '</mn>', '</mrow>'])
        print(vector)
    else:
        model = gensim.models.doc2vec.Doc2Vec.load(checkpoint)

    print("Finished setting up/training the model")

    # # Assessing the Model

    if args.assess:
        print("\nASSESS")
        ranks = []
        second_ranks = []
        assess_num = 10000 # len(train_corpus)
        for doc_id in range(assess_num):
            inferred_vector = model.infer_vector(train_corpus[doc_id].words)
            sims = model.dv.most_similar([inferred_vector], topn=len(model.dv))
            ranked_docs = [docid for docid, sim in sims]
            if trainlist[doc_id] in ranked_docs:
                rank = [docid for docid, sim in sims].index(trainlist[doc_id])
                ranks.append(rank)

                second_ranks.append(sims[1])
            if doc_id % 1000 == 0:
                print(f"Finished {doc_id} out of {len(train_corpus)}")
        
        counter = collections.Counter(ranks)
        print(counter)

        print(trainlist[doc_id])
        print('Document ({}): «{}»\n'.format(doc_id, ' '.join(train_corpus[doc_id].words)))
        print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % model)
        for label, index in [('MOST', 0), ('SECOND-MOST', 1), ('MEDIAN', len(sims)//2), ('LEAST', len(sims) - 1)]:
            print(u'%s %s: «%s»\n' % (label, sims[index], ' '.join(train_corpus[trainlist.index(sims[index][0])].words)))
        
        # Pick a random document from the corpus and infer a vector from the model
        doc_id = random.randint(0, assess_num - 1)

        # Compare and print the second-most-similar document
        print(trainlist[doc_id])
        print('Train Document ({}): «{}»\n'.format(doc_id, ' '.join(train_corpus[doc_id].words)))
        sim_id = second_ranks[doc_id]
        print('Similar Document {}: «{}»\n'.format(sim_id, ' '.join(train_corpus[trainlist.index(sim_id[0])].words)))

        print("Assessed the model")

    # Testing the Model

    if args.test:
        print("\nTEST")
        # Pick a random document from the test corpus and infer a vector from the model
        doc_id = random.randint(0, len(test_corpus) - 1)
        inferred_vector = model.infer_vector(test_corpus[doc_id])
        sims = model.dv.most_similar([inferred_vector], topn=len(model.dv))

        # Compare and print the most/median/least similar documents from the train corpus
        print('Test Document ({}): «{}»\n'.format(doc_id, ' '.join(test_corpus[doc_id])))
        print(testlist[doc_id])
        print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % model)
        for label, index in [('MOST', 0), ('MEDIAN', len(sims)//2), ('LEAST', len(sims) - 1)]:
            print(u'%s %s: «%s»\n' % (label, sims[index], ' '.join(train_corpus[trainlist.index(sims[index][0])].words)))
        
        # Pick 8_0.xml from the test corpus and infer a vector from the model
        doc_id = testlist.index("8")
        inferred_vector = model.infer_vector(test_corpus[doc_id])
        sims = model.dv.most_similar([inferred_vector], topn=len(model.dv))

        # Compare and print the most/median/least similar documents from the train corpus
        print('\nTest Document ({}): «{}»\n'.format(doc_id, ' '.join(test_corpus[doc_id])))
        print(testlist[doc_id])
        print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % model)
        for label, index in [('MOST', 0), ('MEDIAN', len(sims)//2), ('LEAST', len(sims) - 1)]:
            print(u'%s %s: «%s»\n' % (label, sims[index], ' '.join(train_corpus[trainlist.index(sims[index][0])].words)))
        
        print(f"\nTrue Answers to Question {testlist[doc_id]}")
        
        # /home/aw742/Word2vec/data/MSE_dataset_full/dataset_full/math/8/answers/319_34.0_0.xml
        answer_id = trainlist.index("8_319_34")
        index = [docid for docid, sim in sims].index(trainlist[answer_id])
        print(index)
        print(u'%s: «%s»\n' % (sims[index], ' '.join(train_corpus[answer_id].words)))
        # /home/aw742/Word2vec/data/MSE_dataset_full/dataset_full/math/8/answers/931_32.0_0.xml
        answer_id = trainlist.index("8_931_32")
        index = [docid for docid, sim in sims].index(trainlist[answer_id])
        print(index)
        print(u'%s: «%s»\n' % (sims[index], ' '.join(train_corpus[answer_id].words)))
        # /home/aw742/Word2vec/data/MSE_dataset_full/dataset_full/math/8/answers/396100_3.0_0.xml
        answer_id = trainlist.index("8_396100_3")
        index = [docid for docid, sim in sims].index(trainlist[answer_id])
        print(index)
        print(u'%s: «%s»\n' % (sims[index], ' '.join(train_corpus[answer_id].words)))
        # /home/aw742/Word2vec/data/MSE_dataset_full/dataset_full/math/8/answers/3693845_3.0_0.xml
        answer_id = trainlist.index("8_3693845_3")
        index = [docid for docid, sim in sims].index(trainlist[answer_id])
        print(index)
        print(u'%s: «%s»\n' % (sims[index], ' '.join(train_corpus[answer_id].words)))

        # Test random question
        # doc_id = random.randint(0, len(test_corpus) - 1)
        # test_question(doc_id, model, testlist, trainlist, train_corpus)

        print("Tested the model")
    
    if args.eval:
        print("\nEVAL")
        rand_question = random.randrange(len(testlist))
        test_question(rand_question, model, qa_dict, testlist, trainlist, train_corpus)

    if args.metrics:
        print("\nMETRICS")
        if len(args.question) <= 0:
            a_prec = []
            a_rec = []
            mrr = []
            a_dcg = []
            a_idcg = []
            a_ndcg = []
            for q in testlist:
                a_dict = {}
                q_id = testlist.index(q)
                inferred_vector = model.infer_vector(test_corpus[q_id])
                sims = model.dv.most_similar([inferred_vector], topn=len(model.dv))
                for a, sim in sims:
                    a_dict[a] = sim
                a_dict = sorted(a_dict.items(), key=lambda x: x[1], reverse=True)
                ideal_dict = {}
                for answer in qa_dict[q]:
                    rank = int(answer[answer.rfind("_"):])
                    ideal_dict[answer] = rank
                for answer in qa_dict[q]:
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
            if args.question in testlist:
                q_id = testlist.index(args.question)
                inferred_vector = model.infer_vector(test_corpus[q_id])
                sims = model.dv.most_similar([inferred_vector], topn=len(model.dv))
                for a, sim in sims:
                    a_dict[a] = sim
            a_dict = sorted(a_dict.items(), key=lambda x: x[1], reverse=True)
            ideal_dict = {}
            for answer in qa_dict[args.question]:
                rank = int(answer[answer.rfind("_"):])
                ideal_dict[answer] = rank
            for answer in qa_dict[args.question]:
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