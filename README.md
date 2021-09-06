# Learning-Word-Embeddings

## Vocab Stats:

500000it [00:04, 112388.61it/s]
Total number of reviews =  500000
Creating the vocabulary ...
100%|██████████| 500000/500000 [4:05:40<00:00, 33.92it/s]  
100%|██████████| 81210/81210 [00:00<00:00, 1214325.18it/s]
Total size of vocabulary = **81211 81210**
Saving vocabulary files ... 

1000000it [00:08, 119426.26it/s]
Total number of reviews =  1000000
Creating the vocabulary ...
100%|██████████| 1000000/1000000 [3:33:30<00:00, 78.06it/s] 
100%|██████████| 75817/75817 [00:00<00:00, 1342722.04it/s]
Total size of vocabulary = 75818 75817
Saving vocabulary files ... 

finalllllllllllllllllllllllllllllllllllllllllllllllll

1000000it [00:07, 127828.93it/s]
Total number of reviews =  1000000
Creating the vocabulary ...
100%|██████████| 1000000/1000000 [3:29:48<00:00, 79.44it/s]  
100%|██████████| 67829/67829 [00:00<00:00, 1311534.52it/s]
Total size of vocabulary = 67830 67829
Saving vocabulary files ... 



******************** SAMPLING ************************

1000000it [00:07, 125322.59it/s]
Total number of reviews =  1000000
Total corpus words =  119201368
Tokenizing Corpus with subsample= True ...
1000000it [11:00, 1514.58it/s]
Number of Discarded reviews after subsampling =  9339
Generating Co-occurrence Matrix ...
100%|██████████| 990661/990661 [03:07<00:00, 5283.90it/s] 
Learning word vectors using SVD ...


--------------------------
1000000it [00:07, 126840.46it/s]
Total number of reviews =  1000000
Total corpus words =  119201368
Tokenizing Corpus with subsample= False ...
1000000it [07:10, 2320.31it/s]
Number of Discarded reviews after subsampling =  639
Generating Co-occurrence Matrix ...
100%|██████████| 999361/999361 [16:28<00:00, 1010.83it/s]
Learning word vectors using SVD ...



# References:

* Subsampling: https://cs.stackexchange.com/questions/95266/subsampling-of-frequent-words-in-word2vec
    original paper: https://papers.nips.cc/paper/2013/file/9aa42b31882ec039965f3c4923ce901b-Paper.pdf

* Handling sparse matrices: https://www.geeksforgeeks.org/how-to-create-a-sparse-matrix-in-python/

* Replace punctuation with white space: https://stackoverflow.com/questions/34860982/replace-the-punctuation-with-whitespace/34922745

# COocc rrefs:
- https://medium.com/analytics-vidhya/co-occurrence-matrix-singular-value-decomposition-svd-31b3d3deb305
