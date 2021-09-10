# Learning Vector Embeddings for Words

This report describes the methods adopted to train the word embeddings on the *Stanford Amazon Electronics Product Reviews* corpus.

## Analysis for SVD of Co-occurrence based Embeddings:

The dimensionality of each word vector is chosen as **50** and a window size of **5 words** (left and right of the center word) is selected.

### Closest words for the word "Camera":
| | |
:-------------------------:|:-------------------------:
![](./images/camera1.png)  |  ![](./images/camera2.png)

***Observations:***

*  The **custom** model has words such as *photos*, *shoots*, *dslr*, etc. which are semantically very close to the query word "camera".

*  Some exceptions are *opportunities* with the lowest score, indicating that this word occurs several times in similar contexts as the word "camera", and the word *camer* which is a typographical mistake that the model has learnt as well.

### Similar words to a mix of words:

The top most similar (geometrically closest) words for 5 words - adjectives, nouns, verbs combined - are shown below:

***Word : Bought, POS: Verb***

| SVD | CBOW | GENSIM |
:-------------------------:|:-------------------------:|:-------------------------:
![](./images/svd/bought.png)  |  ![](./images/cbow0.001/bought.png)  |  ![](./images/gensim/bought.png)
![](./plots/svd_BOUGHT_custom_model.png)  |  ![](./plots/cbow0.001_BOUGHT_custom_model.png)  |  ![](./plots/svd_BOUGHT_gensim_model.png)

***Word : Broken, POS: Adjective***

| SVD | CBOW | GENSIM |
:-------------------------:|:-------------------------:|:-------------------------:
![](./images/svd/broken.png)  |  ![](./images/cbow0.001/broken.png)  |  ![](./images/gensim/broken.png)
![](./plots/svd_BROKEN_custom_model.png)  |  ![](./plots/cbow0.001_BROKEN_custom_model.png)  |  ![](./plots/svd_BROKEN_gensim_model.png)

***Word : Computer, POS: Noun***

| SVD | CBOW | GENSIM |
:-------------------------:|:-------------------------:|:-------------------------:
![](./images/svd/computer.png)  |  ![](./images/cbow0.001/computer.png)  |  ![](./images/gensim/computer.png)
![](./plots/svd_COMPUTER_custom_model.png)  |  ![](./plots/cbow0.001_COMPUTER_custom_model.png)  |  ![](./plots/svd_COMPUTER_gensim_model.png)

***Word : Excellent, POS: Adjective***

| SVD | CBOW | GENSIM |
:-------------------------:|:-------------------------:|:-------------------------:
![](./images/svd/excellent.png)  |  ![](./images/cbow0.001/excellent.png)  |  ![](./images/gensim/excellent.png)
![](./plots/svd_EXCELLENT_custom_model.png)  |  ![](./plots/cbow0.001_EXCELLENT_custom_model.png)  |  ![](./plots/svd_EXCELLENT_gensim_model.png)

***Word : iPhone, POS: Noun***

| SVD | CBOW | GENSIM |
:-------------------------:|:-------------------------:|:-------------------------:
![](./images/svd/iphone.png)  |  ![](./images/cbow0.001/iphone.png)  |  ![](./images/gensim/iphone.png)
![](./plots/svd_IPHONE_custom_model.png)  |  ![](./plots/cbow0.001_IPHONE_custom_model.png)  |  ![](./plots/svd_IPHONE_gensim_model.png)

***Word : Terrible, POS: Adjective***

| SVD | CBOW | GENSIM |
:-------------------------:|:-------------------------:|:-------------------------:
![](./images/svd/terrible.png)  |  ![](./images/cbow0.001/terrible.png)  |  ![](./images/gensim/terrible.png)
![](./plots/svd_TERRIBLE_custom_model.png)  |  ![](./plots/cbow0.001_TERRIBLE_custom_model.png)  |  ![](./plots/svd_TERRIBLE_gensim_model.png)

## Computational constraints:

* ***Building vocabulary from 1M reviews (instead of 1.6M)***

* ***Training CBOW enc-dec on a smaller sample*** Since the compute time to train 1 epoch of enc-dec CBOW on the entire corpus of 1.6M reviews was nearly 14 hours, to train 20 epochs would take 280 hours which is > 10 days. 10 epochs also would take 140 hours which is ~5 days. Hence the enc-dec CBOW architechture was trained on a smaller subset of 20K reviews. The screenshot of runtime is :

![](./images/cbow_encdec_time.png)