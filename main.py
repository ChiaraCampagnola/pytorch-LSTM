import torch
import torch.nn as nn

from data import TextProcess, train_test_split
from model import TextGenerator, train, generate

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

corpus = TextProcess()

batch_size = 6
_, data_indeces = corpus.get_data('mini_alice.txt')
train_data, test_data = train_test_split(data_indeces)
train_batch = corpus.get_batched_data(train_data, batch_size=batch_size)
test_batch = corpus.get_batched_data(test_data, batch_size=batch_size)
data = (train_batch, test_batch)

# Set parameters
embed_size = 128
hidden_size = 1024
num_layers = 1
epochs = 10
sequence_length = 30
learning_rate = 0.002

vocab_size = len(corpus.dictionary)

model = TextGenerator(vocab_size, embed_size, hidden_size, num_layers)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#model.load_state_dict(torch.load("alice_model_40.pt", map_location=torch.device(device)))

epochs = 2

model.to(device)   
train(model, data, sequence_length, epochs, loss_fn, optimizer)


#torch.save(model.state_dict(), "alice_model_50.pt")

generate(model, corpus, "Once", text_length=100)

