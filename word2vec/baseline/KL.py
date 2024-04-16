import numpy as np
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.feature_extraction.text import CountVectorizer

# formular for P's divergence from Q: KL(P || Q) = – sum x in X P(x) * log(Q(x) / P(x))
# If we are attempting to approximate an unknown probability distribution, then the target.xml probability distribution from data is P, and Q is our approximation of the distribution.

class kl_divergence:
    def __int__(self):
        pass
    def get_common_words(self, question_dict, answerDict):
        common_keys = set(question_dict).intersection(answerDict)
        return common_keys


    def getLDA_Topics(self, question_dict):  # for dataSetStr in range(1) : #dataSetList :
        question_word_list = question_dict.keys()
        numTopics = 0
        # 1/15
        numTopics = (int)(len(question_word_list) /15)
        if numTopics == 0:
            numTopics = 1
        # /2
        # numKeywords = (int)((len(question_word_list)/numTopics)/2)
        numKeywords = (int)(3 + (numTopics / 4))

        bow_vectorizer = CountVectorizer()
        bow_matrix = bow_vectorizer.fit_transform(question_word_list)

        lda_bow = LDA(n_components=numTopics, random_state=42)
        lda_bow.fit(bow_matrix)

        topicList = []
        for idx, topic in enumerate(lda_bow.components_):
            topicList.append([bow_vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-1 * numKeywords:]])

        keywordMap = {}
        # totalMapVal = 0
        for topic in topicList:
            for word in topic:
                # totalMapVal += 1
                if word in keywordMap:
                    keywordMap[word] += 1
                else:
                    keywordMap[word] = 1
        #
        # print("numTopics: ", numTopics)
        # print("numKeywords", numKeywords)

        return topicList, keywordMap, numTopics, numKeywords
        # return topicList, keywordMap, numTopics

    def kl_lda(self, question_dict, questionWordNum, answerDict, answerWordNum, keywordMap, numTopics, numKeywords):
        # print(numTopics,"topic")
        # print(numKeywords, "keywords")
        sum = 0
        common_keys = self.get_common_words(question_dict, answerDict)
        # print(common_keys)
        # print("keywordmap:",keywordMap)
        for word in common_keys:
            if word in keywordMap:
                a =  question_dict[word] /questionWordNum
                b = (answerDict[word] /answerWordNum)
                c = (keywordMap[word] / numTopics)
                val = a*np.log( a / (b*c ) )
                # print(word, np.absolute(val))
                sum+=np.absolute(val)
        # print("sum: ", sum)
        return sum

    def kl_archive_divergence(self, question_dict, questionWordNum, answerDict, answerWordNum):
        sum = 0
        common_keys = self.get_common_words(question_dict, answerDict)
        num_common_q = 0
        for c in common_keys:
            num_common_q+=question_dict[c]
        if len(common_keys) ==0:
            sum = 1
        else:
            for word in common_keys:
                a = question_dict[word] / questionWordNum
                b = (answerDict[word] / answerWordNum)
                val = (a) * np.log( a / b )    #Sophie version
                # print(word, np.absolute(val))
                sum +=np.absolute(val)
        # print("sum score: ", sum)
                # sum +=val
        return sum


    def new_kl_tfidf(self, question_dict, questionWordNum, answer_list, answerDict, answerWordNum):
        sum = 0
        common_keys = self.get_common_words(question_dict, answerDict)
        num_common_q = 0

        for c in common_keys:
            num_common_q += question_dict[c]
        if len(common_keys) == 0:
            sum = 1
        else:
            for word in common_keys:
                a = question_dict[word] / questionWordNum
                b = (answerDict[word] / answerWordNum)
                tf_a = answerDict[word] / max(answerDict.values())
                # print("tfa: ", tf_a)
                tf_q = question_dict[word]/max(question_dict.values())
                # print("tfq: ", tf_q)
                # tf = answerDict[word]
                has_word = 0
                for ans in answer_list:
                    if word in ans.keys():
                        has_word += 1
                if has_word == 0:
                    has_word = 1
                # idf = np.log((len(answer_list)/(1+has_word))+1)
                idf_a = np.log(len(answer_list) / has_word)
                idf_q = np.log(questionWordNum / (max(question_dict.values())-question_dict[word]+1))
                # idf = np.log((len(answer_list)/ has_word)+1)
                if b == 0:
                    continue
                # val = (a) * np.log(a * (tf_q*idf_q) / (b* (tf_a* idf_a)))  # Sophie version
                val = (a) * np.log(a*tf_q / (b* (tf_a* idf_a)))  # Sophie version
                # print(word,np.absolute(val))
                sum += np.absolute(val)
        # print("sum score ", sum)
        return sum

    '''
    arcNumSen: arc number of sentences
    '''
    def KL_TF_IDF(self, question_dict, questionWordNum, answer_list, answerDict):
        sum = 0
        common_keys = self.get_common_words(question_dict, answerDict)
        num_common_q=0

        for c in common_keys:
            num_common_q += question_dict[c]
        if len(common_keys) ==0:
            sum = 1
        else:
            for word in common_keys:
                a = question_dict[word] / questionWordNum
                tf = answerDict[word]/max(answerDict.values())
                # tf = answerDict[word]
                has_word = 0
                for ans in answer_list:
                    if word in ans.keys():
                        has_word+=1
                if has_word==0:
                    has_word=1
                # idf = np.log((len(answer_list)/(1+has_word))+1)
                idf = np.log(len(answer_list)/ has_word)
                # idf = np.log((len(answer_list)/ has_word)+1)
                b = tf*idf
                if b == 0:
                    continue
                val = (a) * np.log(a / b)  # Sophie version
                sum += np.absolute(val)
                # sum += val
        # print("result for the answer: ",sum)
        return sum

        # with open("compare.txt", "a") as filewriter:
        #     filewriter.write("\nquestion: "+str(question_dict.keys())+"\n")
        #     filewriter.write("answer: "+ str(answerDict.keys())+"\n")
        #     filewriter.write("common words: "+ str(common_keys)+"\n")

    #         with open("compare.txt", "a") as filewriter:
    #             filewriter.write("word: "+word+"\n")
    #             filewriter.write("a: "+str(a)+"\n")
    #             filewriter.write("b: "+str(b)+"\n")
    #             filewriter.write("tf: "+str(tf)+"\n")
    #             filewriter.write("idf: "+str(idf)+"\n")
    #             filewriter.write("Sum_{CW}: "+str(val)+"\n\n")
    # with open("compare.txt", "a") as filewriter:
    #     filewriter.write("score: " + str(sum) + "\n")
    #     sum = sum/num_common_q





