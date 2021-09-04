'''
Learning word embeddings using a frequency based approach
by constructing a co-occurence matrix from the given corpus.
'''

import nltk 
import numpy as np
import string
from scipy import sparse
import tqdm
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
from scipy.sparse import csr_matrix
from gensim.parsing.preprocessing import strip_punctuation
import re

class FreqTrain():

    def __init__(self, tokenized_corpus, word2ind, ind2word, num_words) -> None:
        super().__init__()
        self.tokenized_corpus = tokenized_corpus
        self.word2ind = word2ind
        self.ind2word = ind2word
        self.num_words = num_words
        self.comat = None

    def generate_comatrix(self, window_size):

        print("Generating Co-occurrence Matrix ...")
        sparse_comat = {}

        for sentence in tqdm.tqdm(self.tokenized_corpus):
            
            for i, word in enumerate(sentence):
                
                if word not in self.word2ind: #skip words that occur less than 5 times
                    continue
                
                for j in range(max(0,i-window_size), min(len(sentence), i+window_size+1)):

                    if i == j: #ignoring the word itself
                        continue

                    if sentence[j] not in self.word2ind: #skip words that occur less than 5 times
                        continue
                    
                    row = self.word2ind[word]
                    col = self.word2ind[sentence[j]]

                    if (row,col) not in sparse_comat:
                        sparse_comat[(row,col)] = 0
                    sparse_comat[(row,col)]+=1

        self.comat = csr_matrix(
            (
                list(sparse_comat.values()), 
                (
                    [tup[0] for tup in list(sparse_comat.keys())], 
                    [tup[1] for tup in list(sparse_comat.keys())], 
                )
            ),
            shape = (self.num_words, self.num_words)
        )


        np.save('models/cooccurrence_matrix', np.array(self.comat))
        return self.comat


    def perform_svd(self, k):

        print("Learning word vectors using SVD ...")

        n_iters = 10  
        comat_dim_reduced = None        

        svd = TruncatedSVD(n_components=k, n_iter=n_iters, random_state=42)
        comat_dim_reduced = svd.fit_transform(self.comat)

        return comat_dim_reduced

        
