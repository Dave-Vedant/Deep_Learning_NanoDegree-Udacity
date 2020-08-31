def train_rnn(rnn, batch_size, criterion, n_epochs, print_every = 100):
    batch_losses = []
    rnn.train()
    print("Training for %d epochs..." % n_epochs)
    for epoch in range(1,n_epochs+1):
        hidden = rnn.init_hidden(batch_size)

        for batch_i, (inputs, labels) in enumerate(train_loader,1):
            n_batches = len(train_loader.dataset)//batch_size
            if(batch_i > n_batches):
                break

            loss, hidden = forward_back_propagation(rnn,optimizer, criterion, inputs, labels, hidden)
            batch_losses.append(loss)
            
            if batch_i % print_every == 0:
                print('Epoch: {:>4}/{:>4} Loss: {}\n'.format(epoch_i,
                        n_epochs, np.average(batch_)))
            return rnn

        

# supporting function

def forward_back_propagation(rnn, optimizer, criterion, input, target, hidden):
    if(train_on_gpu):
        rnn.cuda()
    
    h = tuple([each.data for each in hidden])
    rnn.zero_grad()

    if(train_on_gpu):
        inputs, target = input.cuda(), target.cuda()
    output, h = rnn(inputs, h)
    loss = criterion(output, target)
    loss.backward()
    nn.utils.clip_grad_norm_(rnn.parameters(),5)
    optimizer.step()
    return loss.item(),h