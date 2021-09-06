'''
Learning word embeddings using a prediction based approach
by using the Continuous Bag of Words (CBOW) model
'''

from numpy.core.defchararray import center
from vocab import Vocab
import nltk 
import numpy as np
import string
from scipy import sparse
import tqdm
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
from scipy.sparse import csr_matrix
import torch
import torch.nn as nn
import torch.utils.data as data_utils

        

class CBOW(torch.nn.Module):

    def __init__(self, vocab_size, embedding_size, hidden_layer_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.W1 = nn.Linear(embedding_size, hidden_layer_size) #input word vector representation
        self.W2 = nn.Linear(hidden_layer_size, vocab_size) #output word vector representation
        self.relu = nn.ReLU()
        self.log_softmax = nn.LogSoftmax(dim = -1)
        
    def forward(self, context_words):

        #print("context words size ", context_words.size())
        embedded_context_words = self.embedding(context_words)  # (batch_size, context_size x embedding_size), 
        #print("embedded words size", embedded_context_words.size())
        output1 = self.W1(embedded_context_words)               # (batch_size, context_size X hidden_layer)
        #print("outputs1 size", output1.size())
        
        mean_output = torch.mean(output1, dim=1)                # (batch_size x hidden_layer_size)
        #print("mean output size ", mean_output.size())

        output2 = self.W2(mean_output)                          # (batch_size x vocab_size)
        #print("outputs2 size", output2.size())
        score_vector = self.log_softmax(output2)
        #print("score_vector_size", score_vector.size())

        return score_vector

class PredTrain():

    def __init__(self, tokenized_corpus, word2ind, ind2word, vocab_size) -> None:
        super().__init__()
        self.tokenized_corpus = tokenized_corpus
        self.word2ind = word2ind
        self.ind2word = ind2word
        self.vocab_size = vocab_size
        self.batch_size = 1024

    
    def generate_context_center_data(self, context_size):

        self.context_center_data = []

        for sentence in tqdm.tqdm(self.tokenized_corpus):
            for i, center_word in enumerate(sentence):
                
                if i < context_size or i >= len(sentence) - context_size:  #to enusre that the batches all have same context
                    continue                                               #possible #TODO: use <unk> tokens if required

                if center_word not in self.word2ind: #skip words that occur less than 5 times
                    continue
                
                context = []
                #aggregating the context words for a given center word
                for j in range(max(0,i-context_size), min(len(sentence), i+context_size+1)):

                    if i == j: #ignoring the word itself
                        continue

                    if sentence[j] not in self.word2ind: #skip words that occur less than 5 times
                        continue

                    context.append(self.word2ind[sentence[j]])

                self.context_center_data.append((
                    torch.tensor(context, dtype=torch.int64),
                    torch.tensor([self.word2ind[center_word]], dtype=torch.int64)
                    ))          

        self.trainloader = torch.utils.data.DataLoader(self.context_center_data, shuffle=True, batch_size=self.batch_size)

    
    def train_cbow(self, embedding_size, hidden_layer_size, model_file_path, embedding_file_path, window_size, num_epochs):

        print("\n----------------------------------------")
        print("Performing CBOW...")
        print("Generating context-center train loader...")
        self.generate_context_center_data(window_size)

        print("Instantiating CBOW Model ...")
        model = CBOW(self.vocab_size, embedding_size, hidden_layer_size)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        
        loss_function = nn.NLLLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.5)

        best_train_loss = 100000
        print("Beginning Training Now ...")

        for epoch in tqdm.tqdm(range(num_epochs)):
            
            train_loss = 0

            for data in self.trainloader:

                context_vectors = data[0].to(device)
                center_vector = data[1].squeeze_().to(device)

                outputs = model(context_vectors)
                loss = loss_function(outputs, center_vector)
                train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            print(f"Train Loss for epoch {epoch} = {train_loss}")
            if train_loss <= best_train_loss:
                torch.save(model, model_file_path)

        torch.save(model.embedding, embedding_file_path)

        print("Quick Testing")
        context = ['The','new','is', 'great']
        context_vector = torch.tensor([self.word2ind[word.lower()] for word in context], dtype=torch.int64).view(1, -1)
        a = model(context_vector)

        #Print result
        print(f'Context: {context}\n')
        print(f'Prediction: {self.ind2word[str(torch.argmax(a[0]).item())]}')






