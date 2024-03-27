import os
import gensim
import collections
import random
import glob
import pathlib
import argparse
from bs4 import BeautifulSoup

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="running doc2vec")
    parser.add_argument("--start-epoch", type=int, default=0)
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--train", type=bool, default=False)
    parser.add_argument("--assess", type=bool, default=False)
    parser.add_argument("--test", type=bool, default=False)
    parser.add_argument("--eval", type=bool, default=False)
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