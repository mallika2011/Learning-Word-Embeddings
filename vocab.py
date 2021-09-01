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


class Vocab():

    def __init__(self, corpus) -> None:
        
        self.corpus = corpus
        self.tokenized_corpus = []
        self.vocabulary = set()
        self.word2ind = {}
        self.ind2word = {}
        self.num_words = 0
        self.lock = threading.Lock()

    def tokenize(self, text):
        text = text.translate(str.maketrans('', '', string.punctuation)).lower()
        textTokens = nltk.word_tokenize(text)

        self.lock.acquire()
        self.vocabulary = self.vocabulary.union(textTokens)
        self.tokenized_corpus.append(textTokens)
        self.lock.release()
    
    def create_vocabulary(self):

        print("Creating the vocabulary ...")
    
        thread_list = []
        #obtain the corpus tokens
        for obj in tqdm.tqdm(self.corpus):
            reviewText = obj["reviewText"]
            thread = threading.Thread(target=self.tokenize, args=(reviewText,))
            thread_list.append(thread)
            thread.start()
            

        print("Collecting threads...")
        for thread in tqdm.tqdm(thread_list):
            thread.join()

        #populate the tally dictionaries
        for token in tqdm.tqdm(self.vocabulary):
            self.word2ind[token] = self.num_words
            self.ind2word[self.num_words] = token
            self.num_words+=1

        print("Total size of vocabulary =", self.num_words + 1)
        print("Saving vocabulary files ... ")

        with open('vocab_files/word2ind.json', 'w') as fp:
            json.dump(self.word2ind, fp)
        with open('vocab_files/ind2word.json', 'w') as fp:
            json.dump(self.ind2word, fp)
        with open("vocab_files/tokenized_corpus.txt", "wb") as fp:
            pickle.dump(self.tokenized_corpus, fp)
        with open('vocab_files/vocabulary.txt','w') as fp:
            fp.write(str(self.vocabulary))
