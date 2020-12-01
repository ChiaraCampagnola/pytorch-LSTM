import torch
import torch.nn as nn

from data import TextProcess
from model import TextGenerator, train, generate

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

corpus = TextProcess()

batch_size = 6
data, data_indeces = corpus.get_data('alice.txt')
batch_data = corpus.get_batched_data(data_indeces, batch_size=batch_size)

# print(f'data: {data}')
# print(f'indeces: {data_indeces}')
# print(batch_data)

# Set parameters
embed_size = 128
hidden_size = 1024
num_layers = 1
epochs = 10
sequence_length = 30
learning_rate = 0.002

vocab_size = len(corpus.dictionary)

# Don't get this
num_batches = batch_data.shape[1] // sequence_length

model = TextGenerator(vocab_size, embed_size, hidden_size, num_layers)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

epochs = 20

#print(model.state_dict()['embed.weight'])

model.load_state_dict(torch.load("alice_model_20epochs.pt"))
        
train(model, batch_data, sequence_length, epochs, loss_fn, optimizer)

#print(model.state_dict()['embed.weight'])

torch.save(model.state_dict(), "alice_model_30.pt")

#model2 = TextGenerator(vocab_size, embed_size, hidden_size, num_layers)
#model2.load_state_dict(torch.load("alice_model.pt"))
        
#print(model2.state_dict()['embed.weight'])

generate(model, corpus, "Alice", text_length=200)


