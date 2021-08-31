'''
Generating the vocabulary of words from the given corpus 
'''

import nltk 
import numpy as np
import string
import tqdm
import json
import pickle

class Vocab():

    def __init__(self, corpus) -> None:
        
        self.corpus = corpus
        self.tokenized_corpus = []
        self.vocabulary = set()
        self.word2ind = {}
        self.ind2word = {}
        self.num_words = 0
    
    def create_vocabulary(self):

        print("Creating the vocabulary ...")

        #obtain the corpus tokens
        for obj in tqdm.tqdm(self.corpus):
            reviewText = obj["reviewText"]
            reviewText = reviewText.translate(str.maketrans('', '', string.punctuation))
            reviewTextTokens = nltk.word_tokenize(reviewText)
            reviewTextTokens = [x.lower() for x in reviewTextTokens]
            self.vocabulary = self.vocabulary.union(reviewTextTokens)
            
            summary = obj["summary"]
            summary = summary.translate(str.maketrans('', '', string.punctuation))
            summaryTokens = nltk.word_tokenize(summary)
            summaryTokens = [x.lower() for x in summaryTokens]
            self.vocabulary = self.vocabulary.union(summaryTokens)

            self.tokenized_corpus.append(reviewTextTokens)
            self.tokenized_corpus.append(summaryTokens)


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
