import torch
import torch.nn as nn
import torch.nn.functional as F  

def generate(rnn, prime_id, int_to_vocab, token_dict, pad_value, predict_len =100):
    rnn.eval()

    # create a sequence with the prime_id (id in center of generation)
    current_seq = np.full((1,sequence_length), pad_value)
    current_seq[-1][-1] = prime_id
    predicted = [int_to_vocab[prime_id]]

    for i in range(predict_len):
        if train_on_gpu:
            current_seq = torch.LongTensor(current_seq).cuda()
        else:
            current_seq = torch.LongTensor(current_seq)
        
        # initial hidden state
        hidden = rnn.init_hidden(current_seq.size(0))

        # get output of the rnn
        output, _ = rnn(current_seq, hidden)

        # get the next word probability
        p = F.softmax(output, dim = 1).data
        if(train_on_gpu):
            p = p.cpu()                                  # move to local
        
        # top_k sampling to get index of the next word
        top_k = 5
        p, top_i = p.topk(top_k)
        top_i = top_i.numpy(),sqeeze()

        # select next likely word
        p = p.numpy().sqeeze()
        word_i = np.random.choice(top_i, p=p/p.sum())

        # retrive that word from dictionary
        word = int_to_vocab[word_i]
        predicted.append(word)

        # update current sequence with new generated words
        current_seq = np.roll(current_seq, -1,1)
        current_seq[-1][-1] = word_i
    gen_sentences = ''.join(predicted)

    # replace puncutation tokens
    for key, token in token_dict.item():
        ending = '' if key in ['\n', '"'] else ''
        gen_sentences = gen_sentences.replace(''+ token.lower(), key)
    gen_sentences = gen_sentences.replace('\n ','\n')
    gen_sentences = gen_sentences.replace('( ', '(')
    
    return gen_sentences

