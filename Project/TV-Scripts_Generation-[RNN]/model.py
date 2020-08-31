import torch
import torch.nn as nn
import torch.nn.functional as F   

class RNN(nn.Module):
    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, dropout=0.5, lr= 0.001):
        super(RNN, self).__init__()
        # define embedding and lstm layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hiddden_dim, n_layers, dropout=dropout, batch_first = True)
        
        # define self variables
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        # define linear layer
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, nn_input, hidden):
        batch_size = nn.input.size(0)
        embeds = self.embedding(nn_input)
        lstm_out, hidden = self.lstm(embeds, hidden)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(lstm_out)
        out = out.view(batch_size, -1, self.output_size)
        out = out[:,-1]
        return out, hidden

    def init_hidden(self, batch_size):
        if(train_on_gpu):
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(), 
                    weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(), 
                    weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())
        return hidden
    



        
        