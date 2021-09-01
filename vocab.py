'''
Generating the vocabulary of words from the given corpus 
'''

import nltk 
import numpy as np
import string
import tqdm
import json
import pickle
import threading
import multiprocessing


class Vocab():

    def __init__(self, corpus) -> None:
        
        self.corpus = corpus
        self.vocabulary = set()
        self.vocabulary_counts = {}
        self.word2ind = {}
        self.ind2word = {}
        self.num_words = 0
        self.lock = threading.Lock()

    def tokenize(self, texts):

        print("Process running ... ")
        for text in tqdm.tqdm(texts):
            text = text.translate(str.maketrans('', '', string.punctuation)).lower()
            textTokens = nltk.word_tokenize(text)
            
            self.lock.acquire()
            for token in textTokens:
                if token not in self.vocabulary_counts:
                    self.vocabulary_counts[token] = 0
                self.vocabulary_counts[token] += 1

            self.vocabulary = self.vocabulary.union(textTokens)
            self.lock.release()
    
    def create_vocabulary(self):

        print("Creating the vocabulary ...")
    
        thread_list = []
        text_list = []
        #obtain the corpus tokens
        for i, obj in tqdm.tqdm(enumerate(self.corpus)):

            reviewText = obj["reviewText"]
            text_list.append(reviewText)

            if (i+1)%100000 == 0:
                thread = multiprocessing.Process(target=self.tokenize, args=(text_list,))
                thread_list.append(thread)
                thread.start()
                text_list.clear()
            

        print("Collecting threads...")
        for thread in tqdm.tqdm(thread_list):
            thread.join()

        #omit the words with less than 5 freqency
        for key, value in self.vocabulary_counts.items():
            if value < 5:
                self.vocabulary.discard(key)

        #populate the tally dictionaries
        for token in tqdm.tqdm(self.vocabulary):
            self.word2ind[token] = self.num_words
            self.ind2word[self.num_words] = token
            self.num_words+=1

        print("Total size of vocabulary =", self.num_words + 1, len(self.vocabulary))
        print("Saving vocabulary files ... ")

        with open('vocab_files/word2ind.json', 'w') as fp:
            json.dump(self.word2ind, fp)
        with open('vocab_files/ind2word.json', 'w') as fp:
            json.dump(self.ind2word, fp)
        with open('vocab_files/vocabulary.txt','w') as fp:
            fp.write(str(self.vocabulary))

