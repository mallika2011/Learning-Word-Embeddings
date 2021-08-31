'''
Run script to:

1. Set up the training corpus and 
prepare the data from the product reviews json 

2. Uses methods of count and prediction to 
train and learn word vector representations 
'''

import json
import sys
import tqdm
from frequency_based_training import *

# Defining key global variables
CORPUS_FILENAME = sys.argv[1]
CORPUS = []



if __name__=='__main__':

    #load and prepare the corpus (1689188 objects)
    with open(CORPUS_FILENAME, 'r') as f:
        for i, line in enumerate(tqdm.tqdm(f)):

            obj = json.loads(line)
            CORPUS.append({
                'reviewText': obj['reviewText'],
                'summary': obj['summary']
            })

    print("Total number of reviews = ", len(CORPUS))

    #train word vectors using the frequency based co-occurence matrix
    freq_train = FreqTrain(CORPUS)
    freq_train.create_vocabulary()
    _ = freq_train.generate_comatrix(3)
    freq_vectors = freq_train.perform_svd(10)



    

