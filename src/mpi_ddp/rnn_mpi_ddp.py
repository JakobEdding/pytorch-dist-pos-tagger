#!/usr/bin/env python
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

from torchtext import data
from torchtext import datasets
from tqdm import tqdm

import time
import random
import sys
import os

# preprocessing
MIN_FREQ = int(os.environ['SUSML_MIN_FREQ'])
# training
RAND_SEED = int(os.environ['SUSML_RAND_SEED'])
NUM_EPOCHS = int(os.environ['SUSML_NUM_EPOCHS'])
BATCH_SIZE = int(os.environ['SUSML_BATCH_SIZE'])
LR = float(os.environ['SUSML_LR'])
# model
RNN_LAYER_TYPE = os.environ['SUSML_RNN_LAYER_TYPE']
# distribution
PARALLELISM_LEVEL = int(os.environ['SUSML_PARALLELISM_LEVEL'])

random.seed(RAND_SEED)
torch.manual_seed(RAND_SEED)
torch.backends.cudnn.deterministic = True

TEXT = data.Field(lower = True)  # can have unknown tokens
UD_TAGS = data.Field(unk_token = None)  # can't have unknown tags

# don't load PTB tags
fields = (("text", TEXT), ("udtags", UD_TAGS), (None, None))

# TODO: how to do this without an internet connection!?
train_data, valid_data, test_data = datasets.UDPOS.splits(fields, root='~/.data')

# inspired by torchtext internals because their splits method is limited to 3 workers... https://github.com/pytorch/text/blob/e70955309ead681f924fecd36d759c37e3fdb1ee/torchtext/data/dataset.py#L325
def custom_split(examples, number_of_parts):
    N = len(examples)
    randperm = random.sample(range(N), len(range(N)))
    indices = [randperm[int(N * (part / number_of_parts)):int(N * (part+1) / number_of_parts)] for part in range(number_of_parts-1)]
    indices.append(randperm[int(N * (number_of_parts-1) / number_of_parts):])
    examples_tuple = tuple([examples[i] for i in index] for index in indices)
    splits = tuple(data.dataset.Dataset(elem, examples.fields) for elem in examples_tuple if elem)
    # In case the parent sort key isn't none
    if examples.sort_key:
        for subset in splits:
            subset.sort_key = examples.sort_key
    return splits

train_data_tuple = custom_split(train_data, PARALLELISM_LEVEL)

# TODO: distribute data...
# print(len(train_data), len(valid_data), len(test_data))

# print(vars(train_data.examples[1800])['text'])
# print(vars(train_data.examples[1800])['udtags'])

TEXT.build_vocab(train_data, min_freq = MIN_FREQ)

UD_TAGS.build_vocab(train_data)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iterators = data.BucketIterator.splits(
    train_data_tuple,
    batch_size = BATCH_SIZE,
    device = device)

valid_iterator, test_iterator = data.BucketIterator.splits(
    (valid_data, test_data),
    batch_size = BATCH_SIZE,
    device = device)

INPUT_DIM = len(TEXT.vocab)

EMBEDDING_DIM = 100
HIDDEN_DIM = 128

OUTPUT_DIM = len(UD_TAGS.vocab)
N_LAYERS = 2
BIDIRECTIONAL = False
DROPOUT = 0.25
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

class BiLSTMPOSTagger(nn.Module):
    def __init__(self,
                 input_dim,
                 embedding_dim,
                 hidden_dim,
                 output_dim,
                 n_layers,
                 bidirectional,
                 dropout,
                 pad_idx):

        super().__init__()

        self.embedding = nn.Embedding(input_dim, embedding_dim, padding_idx = pad_idx)

        if False: # os.environ['RNN_TYPE'] == 'lstm':
            self.rnn = nn.LSTM(embedding_dim,
                            hidden_dim,
                            num_layers = n_layers,
                            bidirectional = bidirectional,
                            dropout = dropout if n_layers > 1 else 0)
        else: # os.environ['RNN_TYPE'] == 'gru':
            self.rnn = nn.GRU(embedding_dim,
                            hidden_dim,
                            num_layers = n_layers,
                            bidirectional = bidirectional,
                            dropout = dropout if n_layers > 1 else 0)
        # else:
        #     raise Exception('has to be lstm or gru')

        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text):

        #text = [sent len, batch size]

        #pass text through embedding layer
        embedded = self.dropout(self.embedding(text))

        #embedded = [sent len, batch size, emb dim]

        #pass embeddings into LSTM
        outputs, (hidden, cell) = self.rnn(embedded)

        #outputs holds the backward and forward hidden states in the final layer
        #hidden and cell are the backward and forward hidden and cell states at the final time-step

        #output = [sent len, batch size, hid dim * n directions]
        #hidden/cell = [n layers * n directions, batch size, hid dim]

        #we use our outputs to make a prediction of what the tag should be
        predictions = self.fc(self.dropout(outputs))

        #predictions = [sent len, batch size, output dim]

        return predictions

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.normal_(param.data, mean = 0, std = 0.1)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def categorical_accuracy(preds, y, tag_pad_idx):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    max_preds = preds.argmax(dim = 1, keepdim = True) # get the index of the max probability
    non_pad_elements = (y != tag_pad_idx).nonzero()
    correct = max_preds[non_pad_elements].squeeze(1).eq(y[non_pad_elements])
    return correct.sum() / torch.FloatTensor([y[non_pad_elements].shape[0]])

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def train(model, iterator, optimizer, criterion, tag_pad_idx, rank, epoch):
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for batch in tqdm(iterator, desc=f'Rank {rank} processing epoch {epoch+1} ...'):
        text = batch.text
        tags = batch.udtags
        optimizer.zero_grad()
        #text = [sent len, batch size]
        predictions = model(text)
        #predictions = [sent len, batch size, output dim]
        #tags = [sent len, batch size]
        predictions = predictions.view(-1, predictions.shape[-1])
        tags = tags.view(-1)
        #predictions = [sent len * batch size, output dim]
        #tags = [sent len * batch size]
        loss = criterion(predictions, tags)
        acc = categorical_accuracy(predictions, tags, tag_pad_idx)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion, tag_pad_idx):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():
        for batch in iterator:
            text = batch.text
            tags = batch.udtags
            predictions = model(text)
            predictions = predictions.view(-1, predictions.shape[-1])
            tags = tags.view(-1)
            loss = criterion(predictions, tags)
            acc = categorical_accuracy(predictions, tags, tag_pad_idx)
            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def run(rank):
    # create local model
    model = BiLSTMPOSTagger(INPUT_DIM,
                        EMBEDDING_DIM,
                        HIDDEN_DIM,
                        OUTPUT_DIM,
                        N_LAYERS,
                        BIDIRECTIONAL,
                        DROPOUT,
                        PAD_IDX)

    model.apply(init_weights)

    # print(f'The model has {count_parameters(model):,} trainable parameters')
    model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)
    TAG_PAD_IDX = UD_TAGS.vocab.stoi[UD_TAGS.pad_token]

    model.to(device)

    print('rank ', rank, ' initial_model: ', sum(parameter.sum() for parameter in model.parameters()))
    # construct DDP model
    ddp_model = DDP(model)
    print('rank ', rank, ' initial_ddp_model: ', sum(parameter.sum() for parameter in ddp_model.parameters()))
    # define loss function and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index = TAG_PAD_IDX)
    criterion = criterion.to(device)
    optimizer = optim.Adam(ddp_model.parameters(),lr=LR)

    best_valid_loss = float('inf')
    overall_start_time = time.time()
    ddp_model.train()

    for epoch in range(NUM_EPOCHS):
        print(f'Starting epoch {epoch+1:02}')
        start_time = time.time()
        train_loss, train_acc = train(ddp_model, train_iterators[rank], optimizer, criterion, TAG_PAD_IDX, rank, epoch)
        valid_loss, valid_acc = evaluate(ddp_model, valid_iterator, criterion, TAG_PAD_IDX)
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(ddp_model.state_dict(), 'tut1-model.pt')

        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')
        # print('rank ', rank, ' parameters: ', sum(parameter.sum() for parameter in ddp_model.parameters()))

    overall_end_time = time.time()
    print('took overall', epoch_time(overall_start_time, overall_end_time))
    ddp_model.load_state_dict(torch.load('tut1-model.pt'))
    test_loss, test_acc = evaluate(ddp_model, test_iterator, criterion, TAG_PAD_IDX)
    print(f'Test Loss: {test_loss:.3f} |  Test Acc: {test_acc*100:.2f}%')

def init_process(fn, backend='mpi'):
    dist.init_process_group(backend)
    fn(dist.get_rank())

if __name__ == "__main__":
    init_process(run)
