'''
Analysis of the word embeddings learnt

- Dimensionality reduction on the word vectors
- Visualizing the closest words to a given set of words
- Finding the top matches to a given word
- Comparing performance with pretrained word vectors
'''

import numpy as np
import json
import sys
import tqdm
from scipy import spatial
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from gensim.test.utils import common_texts
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec


def get_top_10_words_custom_model(query_word):
    vectors = np.load("./freq_word_vectors.npy")

    with open("./vocab_files/word2ind.json", 'r') as f:
        word2ind = json.load(f)
    with open("./vocab_files/ind2word.json", 'r') as f:
        ind2word = json.load(f)

    query_vec = vectors[word2ind[query_word]]
    similarity_scores = []

    for i, vec in tqdm.tqdm(enumerate(vectors)):
        
        result = spatial.distance.cosine(vec, query_vec)
        similarity_scores.append(ind2word[str(i)], result)

    similarity_scores.sort(key = lambda x: x[1])
    top_similar_words = similarity_scores[:10]
    top_similar_vectors = [ vectors[word2ind[word]] for word,_ in top_similar_words]

    return top_similar_words, np.array(top_similar_vectors)

def get_top_10_words_pretrained_model(query_word):
    
    glove_file = datapath('/Users/manning/Corpora/GloVe/glove.6B.100d.txt')
    word2vec_glove_file = get_tmpfile("glove.6B.100d.word2vec.txt")
    
    glove2word2vec(glove_file, word2vec_glove_file)
    gensim_model = KeyedVectors.load_word2vec_format(word2vec_glove_file)

    top_similar_words = gensim_model.most_similar(query_word, topn=10)
    top_similar_vectors = [gensim_model[word] for word, _ in top_similar_words]
    
    return top_similar_words, top_similar_vectors

def plot_closest_words(input_vecs_list, input_words_list, query_word):
    """
    Reduce dimensionality of a an input list of vectors
    Params:
        input_vecs (list of numpy arrays): set of vectors of dimension (num_samples, num_features)
    Return:
        reduced_vecs (list of nump arrays): vectors in reduced dimension vector space
    """

    color_list = ['red']

    for input_vecs, input_words in zip(input_vecs_list, input_words_list):
        reduced_vecs = TSNE(n_components=2).fit_transform(input_vecs)

        fig = plt.figure(figsize=(10,8))
        for point, word in zip(reduced_vecs, input_words):
            x = point[0]
            y = point[1]
            plt.scatter(x, y, marker='o', color='red')
            plt.text(x+0.005, y+0.005, word[0], fontsize=12)
        
        plt.savefig("plots/top_words_for_"+str(query_word)+".png")


if __name__ == '__main__':

    query_words = [
        "broken",
        "computer",
        "bought",
        "excellent",
        "iphone"
    ]

    #obtain and plot the top closest words for these 5 words via gensim and custom model
    print("\n\n********************** Question 1 *************************")
    query_words_top_vectors_custom = []
    query_words_top_words_custom = []
    query_words_top_vectors_gensim = []
    query_words_top_words_gensim = []

    for word in tqdm.tqdm(query_words):

        print("\n--------------------------")
        print("For query word:", word)

        top_closest_words_custom, top_closest_vectors_custom = get_top_10_words_custom_model(word)
        plot_closest_words([top_closest_vectors_custom], [top_closest_words_custom], word)
        query_words_top_vectors_custom.append(top_closest_vectors_custom)
        query_words_top_words_custom.append(top_closest_words_custom)
        print("Closest words via custom model:")
        print(top_closest_words_custom)
        

        top_closest_words_gensim, top_closest_vectors_gensim = get_top_10_words_pretrained_model(word)
        plot_closest_words([top_closest_vectors_gensim], [top_closest_words_gensim], word)
        query_words_top_vectors_gensim.append(top_closest_vectors_gensim)
        query_words_top_words_gensim.append(top_closest_words_gensim)
        print("Closest words via gensim model:")
        print(top_closest_words_gensim)

    plot_closest_words(query_words_top_vectors_custom, query_words_top_words_custom, "combined")
    plot_closest_words(query_words_top_vectors_gensim, query_words_top_words_gensim, "combined")


    #obtain and plot the top closest words for "CAMERA" via gensim and custom model 
    print("\n\n********************** Question 2 *************************")
    report_q2_word = "camera"
    top_closest_words_custom_camera, top_closest_vectors_custom_camera = get_top_10_words_custom_model(word)
    top_closest_words_gensim, top_closest_vectors_gensim = get_top_10_words_pretrained_model(word)
    plot_closest_words([top_closest_vectors_custom_camera], [top_closest_words_custom_camera], "camera")
    plot_closest_words([top_closest_vectors_custom_camera], [top_closest_words_custom_camera], "camera")


    









    
