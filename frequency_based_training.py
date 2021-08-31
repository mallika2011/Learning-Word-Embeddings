'''
Learning word embeddings using a frequency based approach
by constructing a co-occurence matrix from the given corpus.
'''

import nltk 
import numpy as np
import string
import tqdm
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA

class FreqTrain():

    def __init__(self, corpus) -> None:
        super().__init__()
        self.corpus = corpus
        self.tokenized_corpus = []
        self.vocabulary = set()
        self.word2ind = {}
        self.ind2word = {}
        self.num_words = 0
        self.comat = None

    def create_vocabulary(self):

        print("Creating the vocabulary ...")

        #obtain the corpus tokens
        for obj in tqdm.tqdm(self.corpus):
            reviewText = obj["reviewText"]
            reviewText.translate(str.maketrans('', '', string.punctuation))
            reviewTextTokens = nltk.word_tokenize(reviewText)
            self.vocabulary.union(reviewTextTokens)
            
            summary = obj["summary"]
            summary.translate(str.maketrans('', '', string.punctuation))
            summaryTokens = nltk.word_tokenize(summary)
            self.vocabulary.union(summaryTokens)

            self.tokenized_corpus.append(reviewTextTokens)
            self.tokenized_corpus.append(summaryTokens)


        #populate the tally dictionaries
        for token in tqdm.tqdm(self.vocabulary):
            self.word2ind[token] = self.num_words
            self.ind2word[self.num_words] = token
            self.num_words+=1

        print("Total size of vocabulary =", self.num_words + 1)


    def generate_comatrix(self, window_size):

        print("Generating Co-occurrence Matrix ...")

        self.comat = np.zeros((self.num_words+1, self.num_words+1))

        for sentence in tqdm.tqdm(self.tokenized_corpus):
            for i, word in enumerate(sentence):
                for j in range(max(0,i-window_size), min(len(sentence), i+window_size+1)):

                    if i == j: #ignoring the word itself
                        continue

                    self.comat[self.word2ind[word]][self.word2ind[sentence[j]]] += 1

        return self.comat


    def perform_svd(self, k):

        print("Learning word vectors using SVD ...")

        n_iters = 10  
        comat_dim_reduced = None        

        svd = TruncatedSVD(n_components=k, n_iter=n_iters, random_state=42)
        comat_dim_reduced = svd.fit_transform(self.comat)

        return comat_dim_reduced

        