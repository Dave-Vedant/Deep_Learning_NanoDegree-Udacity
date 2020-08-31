'''
helper.py : consist all the supporting function for the main ML flow. 
'''

import os
import pickle
import torch
from torch.utils.data import TensorDataset,DataLoader

special_words = {'PADDING':'<PAD>'}


# load text data
def load_data(path):                       
    input_file = os.path.join(path)
    with open(input_file, "r") as f:
        data = f.read()
    return data

# preprocessing data (Key : Tocken conversion, lower, split) and save data to pickel
def preprocess_data (dataset_path, tocken_lookup, create_lookup_tables):
    text = load_data(dataset_path)
    text = text[81:]

    tocken_dict = tocken_lookup()
    for key, tocken in tocken_dict.items():
        text = text.replace(key, ' {} '.format(tocken))

    text = text.lower()
    text = text.split()

    vocab_to_int, int_to_vocab = create_lookup_tables(text + list(special_words.values()))
    int_word = [vocab_to_int[word] for word in text]
    pickle.dump((int_word, vocab_to_int, int_to_vocab, tocken_dict), open('preprocess.p','wb'))

# load pickle data
def load_process():
    return pickle.load(open('preprocess.p', mode = 'rb'))

# save model 
def save_model(filename, decoder):
    save_filename = os.path.splitext(os.path.basename(filename)[0]) + '.pt'
    torch.save(decoder, save_filename)

# load model 
def load_model(filename):
    load_filename = os.path.splitext(os.path.basename(filename)[0]) + '.pt'
    return torch.load(load_filename)
    
# create lookup table
def create_lookup_tables(text):
    word_counts = Counter(text)
    sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)
    int_to_vocab = {ii: word for ii, word in enumerate(sorted_vocab)}
    vocab_to_int = {word: ii for ii, word in int_to_vocab}
    return (vocab_to_int, int_to_vocab)

def tocken_lookup():
    tokens = dict()
    tokens['.'] = '<PERIOD>'
    tokens[','] = '<COMMA>'
    tokens['"'] = '<QUOTATION_MARK>'
    tokens[';'] = '<SEMICOLON>'
    tokens['!'] = '<EXCLAMATION_MARK>'
    tokens['('] = '<LEFT_PAREN>'
    tokens[')'] = '<RIGHT_PAREN>'
    tokens['?'] = '<QUESTION_MARK>'
    tokens['_'] = '<DASH>'
    tokens['\n'] = '<NEW_LINE>'
    return token

def batch_data(words, sequence_length, batch_size):
    n_batches = words//batch_size
    words = words[n_batches*batch_size]
    y_len = len(words)-sequence_length
    x,y = [], []
    for idx in range(0,y_len):
        idx_end = sequence_length + idx
        x_batch = words[idx:idx_end]
        x.append(x_batch)
        y_batch = words[idx_end]
        y.append(y_batch)

        # create tensor data
        data = TensorDataset(torch.from_numpy(np.asarray(x)), torch.from_numpy(np.asarray(y)))
        data_loader = DataLoader(data, shuffle= False, batch_size = batch_size)
    return data_loader





