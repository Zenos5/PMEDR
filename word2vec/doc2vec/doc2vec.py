import os
import gensim
import collections
import random
import glob
import pathlib
import argparse
import math
import numpy as np
from bs4 import BeautifulSoup
import xlwt 
from xlwt import Workbook 

# Training Corpus: Answers
# Testing Corpus: Questions 
def read_corpus(data_path, qa_dict, testlist, trainlist, tokens_only=False):
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
    p = 0
    dcg = []
    idcg = []
    ndcg = []

    ideal_dict = np.array(ideal_dict)
    # voter_scores = ideal_dict[:, 1]
    voter_scores = [int(i) for i in ideal_dict[:, 1]]
    min_val = np.min(voter_scores)
    #print(type(sims_dict), type(ideal_dict), sims_dict[0], ideal_dict[0])
    # print(ideal_dict, ideal_dict[:, 0])
    for key, _ in sims_dict:
        count += 1
        if key in ideal_dict[:, 0]:
            vote = int(ideal_dict[np.where(ideal_dict[:, 0] == key)[0][0]][1])
            vote = vote - min_val + 1 # subtracts min value to get rid of negative scores, +1 to keep from ranking as irrelevant
            # print(f"Count {count}, Answer {key}, Vote Score {vote}, p {p}")
            if rel_pos == 0:
                rel_pos = count
            if count <= 1:
                p_1 += 1
                dcg.append(vote * 1.0)
            if count <= 3:
                p_3 += 1
            if count <= 5:
                p_5 += 1
            if count <= 10:
                p_10 += 1
            p += 1
            if count > 1:
                dcg.append(dcg[-1] + vote / math.log2(count))
        else:
            if count <= 1:
                dcg.append(0.0)
            else:
                dcg.append(dcg[-1])
    for i in range(len(sims_dict)):
        if ideal_dict.shape[0] >= i + 1:
            if i <= 0:
                idcg.append(int(ideal_dict[i, 1]) * 1.0)
            else:
                idcg.append(idcg[-1] + int(ideal_dict[i, 1]) / math.log2(i + 1))
        else:
            idcg.append(idcg[-1])
    rel_num = ideal_dict.shape[0]
    rr = 0.0
    if rel_pos > 0:
        rr = 1.0 / rel_pos
    precision = [p_1, p_3 / 3.0, p_5 / 5.0, p_10 / 10.0, p / len(sims_dict)]
    recall = [p_1 / rel_num, p_3 / rel_num, p_5 / rel_num, p_10 / rel_num, p / rel_num]
    
    ndcg = [0.0 if idcg[0] == 0.0 else dcg[0] / idcg[0], 
            0.0 if idcg[2] == 0.0 else dcg[2] / idcg[2], 
            0.0 if idcg[4] == 0.0 else dcg[4] / idcg[4], 
            0.0 if idcg[9] == 0.0 else dcg[9] / idcg[9], 
            0.0 if idcg[-1] == 0.0 else dcg[-1] / idcg[-1]]
    p_dcg = [dcg[0], dcg[2], dcg[4], dcg[9], dcg[-1]]
    i_dcg = [idcg[0], idcg[2], idcg[4], idcg[9], idcg[-1]]

    return precision, recall, rr, p_dcg, i_dcg, ndcg

def init_excel_sheet(wb):
    sheet1 = wb.add_sheet('Sheet 1')
    sheet1.write(0, 0, "Question/Metrics")
    sheet1.write(0, 1, "P@1")
    sheet1.write(0, 2, "P@3")
    sheet1.write(0, 3, "P@5")
    sheet1.write(0, 4, "P@10")
    sheet1.write(0, 5, "P@all")
    sheet1.write(0, 6, "R@1")
    sheet1.write(0, 7, "R@3")
    sheet1.write(0, 8, "R@5")
    sheet1.write(0, 9, "R@10")
    sheet1.write(0, 10, "R@all")
    sheet1.write(0, 11, "DCG@1")
    sheet1.write(0, 12, "DCG@3")
    sheet1.write(0, 13, "DCG@5")
    sheet1.write(0, 14, "DCG@10")
    sheet1.write(0, 15, "DCG@all")
    sheet1.write(0, 16, "iDCG@1")
    sheet1.write(0, 17, "iDCG@3")
    sheet1.write(0, 18, "iDCG@5")
    sheet1.write(0, 19, "iDCG@10")
    sheet1.write(0, 20, "iDCG@all")
    sheet1.write(0, 21, "nDCG@1")
    sheet1.write(0, 22, "nDCG@3")
    sheet1.write(0, 23, "nDCG@5")
    sheet1.write(0, 24, "nDCG@10")
    sheet1.write(0, 25, "nDCG@all")
    sheet1.write(0, 26, "RR")
    return sheet1

def write_to_sheet(excel_sheet, index, precision, recall, rr, p_dcg, i_dcg, n_dcg):
    for i in range(5):
        excel_sheet.write(index, i + 1, precision[i])
        excel_sheet.write(index, i + 6, recall[i])
        excel_sheet.write(index, i + 11, p_dcg[i])
        excel_sheet.write(index, i + 16, i_dcg[i])
        excel_sheet.write(index, i + 21, n_dcg[i])
    excel_sheet.write(index, 26, rr)
    return excel_sheet


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="running doc2vec")
    parser.add_argument("--start-epoch", type=int, default=0)
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--train", type=bool, default=False)
    parser.add_argument("--assess", type=bool, default=False)
    parser.add_argument("--test", type=bool, default=False)
    parser.add_argument("--eval", type=bool, default=False)
    parser.add_argument("--metrics", type=bool, default=False)
    parser.add_argument("--question", type=str, default="")
    parser.add_argument("--data-dir", type=str, default="../data/MSE_dataset_full/dataset_full/text/")
    parser.add_argument("--checkpoint", type=str, default="doc2vec/checkpoints/d2v_40.model")
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

    train_corpus = list(read_corpus(data_dir, qa_dict, testlist, trainlist))
    test_corpus = list(read_corpus(data_dir, qa_dict, testlist, trainlist, tokens_only=True))

    print(train_corpus[:2])
    print(test_corpus[:2])

    # Training the Model
    model = None
    if args.train:
        print("\nTRAIN")
        if start_epoch > 0:
            model = gensim.models.doc2vec.Doc2Vec.load(checkpoint)
        else:
            model = gensim.models.doc2vec.Doc2Vec(vector_size=args.vector_size, min_count=args.min_count, dm=args.strategy, negative=args.neg_sample, hs=args.hier_sm, epochs=max_epochs)
            model.build_vocab(train_corpus)

        print(f"Word 'statistics' appeared {model.wv.get_vecattr('statistics', 'count')} times in the training corpus.")

        for epoch in range(start_epoch, max_epochs):
            print('iteration {0}'.format(epoch))
            model.train(train_corpus, total_examples=model.corpus_count, epochs=1)
            if epoch % 5 == 0:
                model.save(f"doc2vec/checkpoints/d2v_{epoch + 1}.model")
                print("Model Saved")

        model.save(f"doc2vec/checkpoints/d2v_{max_epochs}.model")
        print("Model Saved")

        vector = model.infer_vector(['how', 'do', 'you', 'define', 'infinity'])
        print(vector)
    else:
        model = gensim.models.doc2vec.Doc2Vec.load(checkpoint)

    print("Finished setting up/training the model")

    # Assessing the Model

    if args.assess:
        print("\nASSESS")
        ranks = []
        second_ranks = []
        for doc_id in range(len(train_corpus)):
            inferred_vector = model.infer_vector(train_corpus[doc_id].words)
            sims = model.dv.most_similar([inferred_vector], topn=len(model.dv))
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
        doc_id = random.randint(0, len(train_corpus) - 1)

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
        
        # Pick a random document from the test corpus and infer a vector from the model
        doc_id = testlist.index("8")
        inferred_vector = model.infer_vector(test_corpus[doc_id])
        sims = model.dv.most_similar([inferred_vector], topn=len(model.dv))

        # Compare and print the most/median/least similar documents from the train corpus
        print('Test Document ({}): «{}»\n'.format(doc_id, ' '.join(test_corpus[doc_id])))
        print(testlist[doc_id])
        print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % model)
        for label, index in [('MOST', 0), ('MEDIAN', len(sims)//2), ('LEAST', len(sims) - 1)]:
            print(u'%s %s: «%s»\n' % (label, sims[index], ' '.join(train_corpus[trainlist.index(sims[index][0])].words)))
        
        print(f"True Answers to Question {testlist[doc_id]}")

        # /home/aw742/Word2vec/data/MSE_dataset_full/dataset_full/text/8/answers/17_28.0.txt
        answer_id = trainlist.index("8_17_28")
        index = [docid for docid, sim in sims].index(trainlist[answer_id])
        print(index)
        print(u'%s: «%s»\n' % (sims[index], ' '.join(train_corpus[answer_id].words)))
        # /home/aw742/Word2vec/data/MSE_dataset_full/dataset_full/text/8/answers/24_5.0.txt
        answer_id = trainlist.index("8_24_5")
        index = [docid for docid, sim in sims].index(trainlist[answer_id])
        print(index)
        print(u'%s: «%s»\n' % (sims[index], ' '.join(train_corpus[answer_id].words)))
        # /home/aw742/Word2vec/data/MSE_dataset_full/dataset_full/text/8/answers/26_4.0.txt
        answer_id = trainlist.index("8_26_4")
        index = [docid for docid, sim in sims].index(trainlist[answer_id])
        print(index)
        print(u'%s: «%s»\n' % (sims[index], ' '.join(train_corpus[answer_id].words)))
        # /home/aw742/Word2vec/data/MSE_dataset_full/dataset_full/text/8/answers/319_34.0.txt
        answer_id = trainlist.index("8_319_34")
        index = [docid for docid, sim in sims].index(trainlist[answer_id])
        print(index)
        print(u'%s: «%s»\n' % (sims[index], ' '.join(train_corpus[answer_id].words)))
        # /home/aw742/Word2vec/data/MSE_dataset_full/dataset_full/text/8/answers/931_32.0.txt
        answer_id = trainlist.index("8_931_32")
        index = [docid for docid, sim in sims].index(trainlist[answer_id])
        print(index)
        print(u'%s: «%s»\n' % (sims[index], ' '.join(train_corpus[answer_id].words)))
        # /home/aw742/Word2vec/data/MSE_dataset_full/dataset_full/text/8/answers/934_4.0.txt
        answer_id = trainlist.index("8_934_4")
        index = [docid for docid, sim in sims].index(trainlist[answer_id])
        print(index)
        print(u'%s: «%s»\n' % (sims[index], ' '.join(train_corpus[answer_id].words)))
        # /home/aw742/Word2vec/data/MSE_dataset_full/dataset_full/text/8/answers/396100_3.0.txt
        answer_id = trainlist.index("8_396100_3")
        index = [docid for docid, sim in sims].index(trainlist[answer_id])
        print(index)
        print(u'%s: «%s»\n' % (sims[index], ' '.join(train_corpus[answer_id].words)))
        # /home/aw742/Word2vec/data/MSE_dataset_full/dataset_full/text/8/answers/3693845_3.0.txt
        answer_id = trainlist.index("8_3693845_3")
        index = [docid for docid, sim in sims].index(trainlist[answer_id])
        print(index)
        print(u'%s: «%s»\n' % (sims[index], ' '.join(train_corpus[answer_id].words)))

        print("Tested the model")
    
    if args.eval:
        print("\nEVAL")
        rand_question = random.randrange(len(testlist))
        test_question(rand_question, model, qa_dict, testlist, trainlist, train_corpus)
    
    if args.metrics:
        print("\nMETRICS")
        if len(args.question) <= 0:
            index = 0
            wb = Workbook()
            excel_sheet = init_excel_sheet(wb)
            a_prec = []
            a_rec = []
            mrr = []
            a_dcg = []
            a_idcg = []
            a_ndcg = []
            for q in testlist:
                index += 1
                excel_sheet.write(index, 0, q)
                a_dict = {}
                q_id = testlist.index(q)
                inferred_vector = model.infer_vector(test_corpus[q_id])
                sims = model.dv.most_similar([inferred_vector], topn=len(model.dv))
                for a, sim in sims:
                    a_dict[a] = sim
                a_dict = sorted(a_dict.items(), key=lambda x: x[1], reverse=True)
                ideal_dict = {}
                for answer in qa_dict[q]:
                    rank = int(answer[answer.rfind("_")+1:])
                    ideal_dict[answer] = rank
                ideal_dict = sorted(ideal_dict.items(), key=lambda x: x[1], reverse=True)
                precision, recall, rr, p_dcg, i_dcg, n_dcg = calc_metrics(a_dict, ideal_dict)
                excel_sheet = write_to_sheet(excel_sheet, index, precision, recall, rr, p_dcg, i_dcg, n_dcg)
                a_prec.append(precision)
                a_rec.append(recall)
                mrr.append(rr)
                a_dcg.append(p_dcg)
                a_idcg.append(i_dcg)
                a_ndcg.append(n_dcg)
            
            wb.save('d2v_metrics_allQA.xls')

            print("\nMetrics for full question set")
            print("\nP@1 P@3 P@5 P@10 P@all")
            print(np.mean(a_prec, axis=0))
            print("\nR@1 R@3 R@5 R@10 R@all")
            print(np.mean(a_rec, axis=0))
            print("\nMRR:", np.mean(mrr))
            print("\nDCG@1 DCG@3 DCG@5 DCG@10 DCG@all")
            print(np.mean(a_dcg, axis=0))
            print("\niDCG@1 iDCG@3 iDCG@5 iDCG@10 iDCG@all")
            print(np.mean(a_idcg, axis=0))
            print("\nnDCG@1 nDCG@3 nDCG@5 nDCG@10 nDCG@all")
            print(np.mean(a_ndcg, axis=0))
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
                rank = int(answer[answer.rfind("_")+1:])
                ideal_dict[answer] = rank
            ideal_dict = sorted(ideal_dict.items(), key=lambda x: x[1], reverse=True)
            precision, recall, rr, p_dcg, i_dcg, n_dcg = calc_metrics(a_dict, ideal_dict)

            print("\nMetrics for question", args.question)
            print("\nP@1 P@3 P@5 P@10 P@all")
            print(precision)
            print("\nR@1 R@3 R@5 R@10 R@all")
            print(recall)
            print("\nRR:", rr)
            print("\nDCG@1 DCG@3 DCG@5 DCG@10 DCG@all")
            print(p_dcg)
            print("\niDCG@1 iDCG@3 iDCG@5 iDCG@10 iDCG@all")
            print(i_dcg)
            print("\nnDCG@1 nDCG@3 nDCG@5 nDCG@10 nDCG@all")
            print(n_dcg)