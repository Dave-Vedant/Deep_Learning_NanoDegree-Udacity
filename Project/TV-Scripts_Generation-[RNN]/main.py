import helper
import train
import project_unitTest
import model

import numpy as np  
import pandas as pd  
import torch  
import matplotlib.pyplot
# %matplotlib inline 
import pickle
import glob


# load data
data_dir = './data/Seinfeld_Script.txt'
text = helper.load_data(data_dir)

# preprocess data
helper.preprocess_data(data_dir, tocken_lookup, create_lookup_tables)
int_text, vocab_to_int, int_to_vocab, token_dict = helper.load_process()

# check gpu
train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('No GPU found, check the availability for another gpu')

# parameters 
# Training Parameters
sequence_length = 10
batch_size = 128
from helper import batch_data
train_loader = batch_data(int_text, sequence_length, batch_size)
num_epochs = 10
learning_rate = 0.001

# Model Parameters
vocab_size = len(vocab_to_int)
output_size = vocab_size
embedding_dim = 200
hidden_dim = 250
n_layers = 2

# show stat for every n number of batches
print_every = 2000

rnn = RNN(vocab_size, output_size, embedding_dim, hidden_dim, n_layers, dropout = 0.5)
if train_on_gpu:
    rnn.cuda()

optimizer = torch.optim.Adam(rnn.parameters(), lr = learning_rate)

# training  the model 
trained_rnn = train_rnn(rnn, batch_size, optimizer, criterion, num_epochs, print_every)

helper.save_model('./save/trained_rnn')
print('Model Trained and Saved')

# =================== Wait for Reply ===================== # 



