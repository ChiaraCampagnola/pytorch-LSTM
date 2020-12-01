import torch

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        # If the word doesn't already exist, add it
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __len__(self):
        return len(self.word2idx)

class TextProcess(object):
    def __init__(self):
        self.dictionary = Dictionary()

    def get_data(self, path):
        data = []
        with open(path, 'r') as file:
            for line in file:
                words = line.split() + ['<EOL>'] # Separate words and add end of line token
                for word in words:
                    self.dictionary.add_word(word)
                    data.append(word)

        # Create tensor containing all indeces of the tokens
        num_tokens = len(data)
        data_idx = torch.LongTensor(num_tokens)
        for index, word in enumerate(data):
            data_idx[index] = self.dictionary.word2idx[word]
        return data, data_idx
    
    def get_batched_data(self, data_indeces, batch_size = 10):
        num_batches = data_indeces.shape[0] // batch_size
        max_num_tokens = num_batches*batch_size
        batch_data = data_indeces[:max_num_tokens]
        batch_data = batch_data.view(batch_size, -1)
        return batch_data
