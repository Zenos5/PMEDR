import nltk
import numpy as np
#import random
import string
import urllib.request
import re
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.stem.porter import *
from collections import Counter
'''
question_str= "In general, how can I evaluate the following sum formula: I am more concerned with how I can derive that answer using Taylor series. It is convergent, but my class has never learned these before. so I feel that there must be a simpler method."
answer_str = "No need to use Taylor series. This can be derived in a similar way to the formula for geometric series. Let's find a general formula for the following sum"
#remove special characters
'''
class clean:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))

    def remove_stop(self, text):
        # text = text.strip()
        # print("text",text)
        tokens=[re.sub(r"[^a-zA-Z0-9]+", '', k) for k in text.split(' ')]
        # print("tokens",tokens)
        #remove stop words
        tokens = [w.lower() for w in tokens if not w.lower() in self.stop_words]
        while '' in tokens:
            tokens.remove('')
        return tokens

    def stem(self, text):
        # stemming
        stemmer = PorterStemmer()
        stemmed = [stemmer.stem(w) for w in text]
        return stemmed

    def indexing(self, text):
        index = Counter(text)
        return index

if __name__ == "__main__":
    s = set(stopwords.words('english'))
    print(s)
    if "sum" in set(stopwords.words('english')):
        print("yes")

