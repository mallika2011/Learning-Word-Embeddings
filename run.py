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
from vocab import *
import random
import numpy as np

# Defining key global variables
CORPUS_FILENAME = sys.argv[1]
CORPUS = []
RUN_TYPE = sys.argv[2] #0:create vocabulary, 1: frequency based, 2: prediction based
global word2ind
global ind2word
global word2count
global vocabulary

def get_sampling_proability(word):

    t = 1e-5 #heuristcally chosen parameter from the original paper by Mikolov et. al 

    total_corpus_words = sum(word2count.values())
    word_count = word2count[word]
    f = word_count/total_corpus_words
    prob = 1 - np.sqrt(t/f)

    return random.random() < prob
    

def tokenize_corpus(subsample):

    tokenized_corpus = []
    print("Tokenizing Corpus ...")

    for obj in tqdm.tqdm(CORPUS):

        tokenized_sentence = []
        reviewText = obj["reviewText"]
        reviewText = reviewText.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation))).lower()
        sentence = nltk.word_tokenize(reviewText)

        if subsample:
            for word in sentence:
                if not get_sampling_proability(word):
                    tokenized_sentence.append(word)

        else:
            tokenized_sentence = sentence
        
        if tokenized_sentence:
            tokenized_corpus.append(tokenized_sentence)

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

        load_corpus()
        tokenized_corpus = tokenize_corpus(subsample=True)
        #train word vectors using the frequency based co-occurence matrix
        window_size = 5
        vector_dim = 50 

        word2ind, ind2word, word2count, vocabulary = load_vocabulary()
        freq_train = FreqTrain(tokenized_corpus, word2ind, ind2word, len(vocabulary))
        _ = freq_train.generate_comatrix(window_size)
        freq_vectors = freq_train.perform_svd(vector_dim)
        np.save('freq_word_vectors', np.array(freq_vectors))

    elif RUN_TYPE == "2":
        print("Not implemented yet ...")



    

