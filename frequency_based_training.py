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

    def __init__(self, corpus, word2ind, ind2word, num_words) -> None:
        super().__init__()
        self.corpus = corpus
        self.word2ind = word2ind
        self.ind2word = ind2word
        self.num_words = num_words
        self.comat = None

    def generate_comatrix(self, window_size):

        print("Generating Co-occurrence Matrix ...")

        self.comat = np.zeros((self.num_words+1, self.num_words+1))

        for obj in tqdm.tqdm(self.corpus):
            sentence = obj["reviewText"].translate(str.maketrans('', '', string.punctuation)).lower()
            sentence = nltk.word_tokenize(sentence)
            
            for i, word in enumerate(sentence):
                for j in range(max(0,i-window_size), min(len(sentence), i+window_size+1)):

                    if i == j: #ignoring the word itself
                        continue

                    self.comat[self.word2ind[word]][self.word2ind[sentence[j]]] += 1

        np.save('cooccurrence_matrix', np.array(self.comat))
        return self.comat


    def perform_svd(self, k):

        print("Learning word vectors using SVD ...")

        n_iters = 10  
        comat_dim_reduced = None        

        svd = TruncatedSVD(n_components=k, n_iter=n_iters, random_state=42)
        comat_dim_reduced = svd.fit_transform(self.comat)

        return comat_dim_reduced

        
