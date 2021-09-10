'''
Run script to:

1. Set up the training corpus and 
prepare the data from the product reviews json 

2. Uses methods of count and prediction to 
train and learn word vector representations 
'''

import json
import sys
import ast
import tqdm
from frequency_based_training import *
from prediction_based_training_nn_based import *
# from prediction_based_training_encdec_based import *
from vocab import *
import random
import numpy as np
import torch
import torch.nn as nn

# Defining key global variables
CORPUS_FILENAME = sys.argv[1]
CORPUS = []
RUN_TYPE = sys.argv[2] #0:create vocabulary, 1: frequency based, 2: prediction based
word2ind = {}
ind2word = {}
word2count = {}
vocabulary = set()
TOTAL_CORPUS_WORDS = 0

def get_sampling_proability(word):

    global TOTAL_CORPUS_WORDS
    t = 1e-5 #heuristcally chosen parameter from the original paper by Mikolov et. al 
    word_count = word2count[word]
    f = word_count/TOTAL_CORPUS_WORDS
    prob = 1 - np.sqrt(t/f)
    
    return np.random.rand() < prob
    

def tokenize_corpus(subsample):

    discarded_reviews = 0
    tokenized_corpus = []
    global TOTAL_CORPUS_WORDS 
    global word2ind
    global word2count
    TOTAL_CORPUS_WORDS = sum(word2count.values())

    print("Total corpus words = ", TOTAL_CORPUS_WORDS)
    print("Tokenizing Corpus with subsample=", subsample, "...")

    for i, obj in tqdm.tqdm(enumerate(CORPUS)):

        tokenized_sentence = []
        reviewText = obj["reviewText"]
        reviewText = reviewText.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation))).lower()
        sentence = nltk.word_tokenize(reviewText)

        for word in sentence:
            if word not in word2ind: #skipping words that are not in the vocabulary (either < 5 occurence, or UNK)
                continue

            if subsample:
                if not get_sampling_proability(word):
                    tokenized_sentence.append(word)
            else:
                tokenized_sentence.append(word)
        
        if tokenized_sentence:
            tokenized_corpus.append(tokenized_sentence)
        else:
            discarded_reviews += 1


    print("Number of Discarded reviews after subsampling = ", discarded_reviews)

    return tokenized_corpus


def load_corpus():
    #load and prepare the corpus (1689188 objects)
    with open(CORPUS_FILENAME, 'r') as f:
        for i, line in enumerate(tqdm.tqdm(f)):

            #consider first 1M reviews - due to computation limitations
            if i >= 1000000:
                break
            
            obj = json.loads(line)
            CORPUS.append({
                'reviewText': obj['reviewText'],
                #'summary': obj['summary']
            })

    print("Total number of reviews = ", len(CORPUS))

def load_vocabulary():
    global word2ind
    global ind2word
    global word2count
    global vocabulary
    with open('vocab_files/word2ind.json', 'r') as fp:
        word2ind = json.load(fp)
    with open('vocab_files/ind2word.json', 'r') as fp:
        ind2word = json.load(fp)
    with open('vocab_files/word2count.json', 'r') as fp:
        word2count = json.load(fp)   
    with open('vocab_files/vocabulary.txt','r') as fp:
        vocabulary = ast.literal_eval(fp.read())

    return word2ind, ind2word, word2count, vocabulary

if __name__=='__main__':

    if RUN_TYPE == "0":

        #load and prepare the corpus (1689188 objects)
        load_corpus()
        #create the vocabulary and corresponding token ID-word mappings
        vocab = Vocab(CORPUS)
        vocab.create_vocabulary()


    elif RUN_TYPE == "1":

        word2ind, ind2word, word2count, vocabulary = load_vocabulary()
        with open('vocab_files/tokenized_corpus_with_subsample.txt', 'rb') as f:
           tokenized_corpus = pickle.load(f)

        #train word vectors using the frequency based co-occurence matrix
        window_size = 5
        vector_dim = 50 

        freq_train = FreqTrain(tokenized_corpus, word2ind, ind2word, len(vocabulary))
        _ = freq_train.generate_comatrix(window_size)
        freq_vectors = freq_train.perform_svd(vector_dim)
        np.save('./embeddings/freq_word_vectors', np.array(freq_vectors))

    elif RUN_TYPE == "2":

        '''
        Parameters to be passed for enc-dec are:
            embedding_size=100, 
            hidden_layer_size=100, 
            model_file_path="./models/cbow_model_v2.pt", 
            embedding_file_path="./embeddings/cbow_embeddings.pt",
            window_size=5, 
            num_epochs=20
        ''' 
        

        word2ind, ind2word, word2count, vocabulary = load_vocabulary()
        with open('vocab_files/tokenized_corpus_with_subsample.txt', 'rb') as f:
           tokenized_corpus = pickle.load(f)


        pred_train = PredTrain(tokenized_corpus, word2ind, ind2word, len(vocabulary))
        pred_train.train_cbow(
            embedding_size=50, 
            hidden_layer_size=128, 
            model_file_path="./models/cbow_model.pt", 
            embedding_file_path="./embeddings/cbow_embeddings.pt",
            window_size=5, 
            num_epochs=20
            )

    elif RUN_TYPE == "3":

        load_corpus()
        word2ind, ind2word, word2count, vocabulary = load_vocabulary()
        tokenized_corpus = tokenize_corpus(subsample=True)
        with open('vocab_files/tokenized_corpus_with_subsample.txt', 'wb') as f:
            pickle.dump(tokenized_corpus, f)

        tokenized_corpus = tokenize_corpus(subsample=False)
        with open('vocab_files/tokenized_corpus_without_subsample.txt', 'wb') as f:
            pickle.dump(tokenized_corpus, f)
        
        



    

