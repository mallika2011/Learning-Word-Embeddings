import numpy as np
import json
from scipy import spatial
import sys
import tqdm
import matplotlib.pyplot as plt

def get_closest_words(embedding_file):

    vectors = np.load(embedding_file)

    with open("./vocab_files/word2ind.json", 'r') as f:
        word2ind = json.load(f)

    with open("./vocab_files/ind2word.json", 'r') as f:
        ind2word = json.load(f)

    query_term = sys.argv[1]
    query_vec = vectors[word2ind[query_term]]

    similarity_scores = {}

    for i, vec in tqdm.tqdm(enumerate(vectors)):
        
        result = spatial.distance.cosine(vec, query_vec)
        similarity_scores[ind2word[str(i)]] = result


    similarity_scores = {k: v for k, v in sorted(similarity_scores.items(), key=lambda item: item[1])}

    print("Top similar words to", query_term, "are:")

    for word in list(similarity_scores.items())[:10]:
        print(word)

def plot_train_graphs():

    cbow_001_loss = [
        9.068363415475737,
        8.63193149990893,
        8.414806427859737,
        8.272358473011229,
        8.17123854094929,
        8.094932057667185,
        8.03451549796198,
        7.985142821440386,
        7.944094619251183,
        7.90925399258978,
        7.879395892168239,
        7.853435440080911,
        7.83092137461098,
        7.810950032971677,
        7.793041866047945,
        7.777229709061524,
        7.762770000134642,
        7.749775461857091,
        7.737859968584097,
        7.726908946603918
    ]
    cbow_enc_dec_loss = [
        9.450149525457354,
        9.07438833322098,
        8.95063620538854,
        8.683043049342597,
        8.324364295646326,
        7.897142673606303,
        7.406941106070334,
        6.884162292551639,
        6.3914515011346165,
        5.963956834664986,
        5.612478620970427,
        5.337806141198571,
        5.108808795017983,
        4.909369954422338,
        4.734135812787867,
        4.570291816298641,
        4.410382422048654,
        4.2626311850191945,
        4.113636223237906,
        3.9723784576601058
    ]

    plt.plot(np.arange(0, 20), cbow_001_loss, label="nn.embedding Based Method")
    plt.plot(np.arange(0, 20), cbow_enc_dec_loss, label="Enc-Dec based method")
    plt.title("Training loss for CBOW")
    plt.legend()
    plt.savefig("./plots/cbow_loss.png")


# get_closest_words(sys.argv[1])
plot_train_graphs()