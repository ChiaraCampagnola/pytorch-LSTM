import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import math

import numpy as np

class TextGenerator(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(TextGenerator, self).__init__() # Why do I need this? What initialisation is happening in nn.Module?
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first = True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, h):
        
        # x shape: (num_batches, seq_length)
        x = self.embed(x) # Shape: (num_batches, seq_length, embed_size)
        out, (h, c) = self.lstm(x, h)
        # out = output features (h_t) of last layer of LSTM (so ALL the final h_ts for all units)
        # out shape: (batch, seq_len, num_directions * hidden_size)

        # input to linear layer is (num_samples, hidden_size), so reshape out:
        num_batches, seq_length, hidden_size = out.shape
        out = out.reshape(num_batches*seq_length, hidden_size)
        out = self.linear(out)
        return out, (h, c)

def train(model, batch_data, seq_length, epochs, loss_function, optimizer):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Training model...')

    batch_size, num_batches = batch_data.shape
    hidden_size = model.lstm.hidden_size
    num_layers = model.lstm.num_layers
    
    for epoch in range(epochs):
        model.train()
        
        # Initialise hidden and cell state
        h = torch.zeros(num_layers, batch_size, hidden_size)
        c = torch.zeros(num_layers, batch_size, hidden_size)
        h = h.to(device)
        c = c.to(device)
        
        epoch_loss = 0.0

        num_steps = math.ceil((num_batches-seq_length)/seq_length)
        current_step = 1
        for i in range(0, num_batches-seq_length, seq_length):
            src = batch_data[:, i:i+seq_length]
            trg = batch_data[:, (i+1):(i+1)+seq_length]
            
            src = src.to(device)
            trg = trg.to(device)
            
            out, _ = model(src, (h,c))          
            loss = loss_function(out, trg.reshape(-1))
            model.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            
            epoch_loss += loss.item()
            print(f'Running epoch [{epoch+1}/{epochs}],'
                  f'Step: [{current_step}/{num_steps}],'
                  f'Loss: {epoch_loss:.2f}               ', end="\r", flush=True)
            current_step += 1
            
        
        print(f'Epoch [{epoch+1}/{epochs}], Train loss: {epoch_loss:.2f}                                ')
    print("Finished training\n")
        
def generate(model, corpus, first_word, text_length, path='result.txt'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Generating text...")
    
    hidden_size = model.lstm.hidden_size
    num_layers = model.lstm.num_layers
    with torch.no_grad():
        with open(path, 'w') as file:
            file.write(first_word+' ')
            
            # Set initial hidden states            
            h = torch.zeros(num_layers, 1, hidden_size)
            c = torch.zeros(num_layers, 1, hidden_size)
            h = h.to(device)
            c = c.to(device)
            
            # Get the representation of the first word:
            first_word_idx = corpus.dictionary.word2idx[first_word]
            input = torch.tensor([first_word_idx]).long().unsqueeze(1)
            input = input.to(device)

            for i in range(text_length):
                output, _ = model(input, (h,c))
                
                # Sample from output
                # Get top 20 words:
                output = output.exp()
                top_probs, top_indeces = torch.topk(output, 20)
                top_probs = top_probs.squeeze()
                top_indeces = top_indeces.squeeze()
                
                #next_word = torch.multinomial(output.exp(), num_samples=1).item()
                next_word = torch.multinomial(top_probs, num_samples=1).item()
                next_word = top_indeces[next_word].item()
                input.fill_(next_word)
                
                # Add sampled word to file
                word = corpus.dictionary.idx2word[next_word]
                if word not in ["'", ",", "!", "\n", ".", ";", "?", ":", '"']:
                    word = ' ' + word
                file.write(word)
            
                if (i+1)%100 == 0:
                    print(f'[{i+1}/{text_length}] words generated.', end="\r", flush=True)
        print(f'[{i+1}/{text_length}] words generated and saved to {path}.')
