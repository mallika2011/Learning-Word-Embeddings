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

# Defining key global variables
CORPUS_FILENAME = sys.argv[1]
CORPUS = []
RUN_TYPE = sys.argv[2] #0:create vocabulary, 1: frequency based, 2: prediction based


def load_corpus():
    #load and prepare the corpus (1689188 objects)
    with open(CORPUS_FILENAME, 'r') as f:
        for i, line in enumerate(tqdm.tqdm(f)):

            #consider first 500K reviews - due to computation limitations
            if i >= 1000000:
                break
            
            obj = json.loads(line)
            CORPUS.append({
                'reviewText': obj['reviewText'],
                #'summary': obj['summary']
            })

    print("Total number of reviews = ", len(CORPUS))

def load_vocabulary():
    with open('vocab_files/word2ind.json', 'r') as fp:
        word2ind = json.load(fp)
    with open('vocab_files/ind2word.json', 'r') as fp:
        ind2word = json.load(fp)
    with open('vocab_files/vocabulary.txt','r') as fp:
        vocabulary = ast.literal_eval(fp.read())

    return word2ind, ind2word, vocabulary

if __name__=='__main__':

    if RUN_TYPE == "0":

        #load and prepare the corpus (1689188 objects)
        load_corpus()
        #create the vocabulary and corresponding token ID-word mappings
        vocab = Vocab(CORPUS)
        vocab.create_vocabulary()


    elif RUN_TYPE == "1":

        load_corpus()
        #train word vectors using the frequency based co-occurence matrix
        window_size = 5
        vector_dim = 50 

        word2ind, ind2word, vocabulary = load_vocabulary()
        freq_train = FreqTrain(CORPUS, word2ind, ind2word, len(vocabulary))
        _ = freq_train.generate_comatrix(window_size)
        freq_vectors = freq_train.perform_svd(vector_dim)
        np.save('freq_word_vectors', np.array(freq_vectors))

    elif RUN_TYPE == "2":
        print("Not implemented yet ...")



    

