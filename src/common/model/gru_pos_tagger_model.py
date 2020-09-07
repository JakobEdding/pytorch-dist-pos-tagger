import time
import random
import sys
import os
from datetime import datetime

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP


# preprocessing
MIN_FREQ = int(os.environ['SUSML_MIN_FREQ'])
# training
RAND_SEED = int(os.environ['SUSML_RAND_SEED'])
NUM_EPOCHS = int(os.environ['SUSML_NUM_EPOCHS'])
BATCH_SIZE = int(os.environ['SUSML_BATCH_SIZE'])
LR = float(os.environ['SUSML_LR'])
# distribution
PARALLELISM_LEVEL = int(os.environ['SUSML_PARALLELISM_LEVEL'])

print(f'RAND-TEST rand seed in gru_pos_tagger_model.py is {RAND_SEED}')
random.seed(RAND_SEED)
torch.manual_seed(RAND_SEED)
torch.backends.cudnn.deterministic = True

class GRUPOSTaggerModel(nn.Module):
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

        self.rnn = nn.GRU(embedding_dim,
                        hidden_dim,
                        num_layers = n_layers,
                        bidirectional = bidirectional,
                        dropout = dropout if n_layers > 1 else 0)

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
