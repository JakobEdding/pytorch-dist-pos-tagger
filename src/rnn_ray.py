#!/usr/bin/env python
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

import ray

import numpy as np

from bilstm_pos_tagger import BiLSTMPOSTagger

from torchtext import data
from torchtext import datasets
from tqdm import tqdm

import time
import random
import sys
import os
from datetime import datetime

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
print('parallelism level is', PARALLELISM_LEVEL)

random.seed(RAND_SEED)
torch.manual_seed(RAND_SEED)
torch.backends.cudnn.deterministic = True



@ray.remote
class ParameterServer(object):
    def __init__(self):
        TEXT = data.Field(lower = True)  # can have unknown tokens
        UD_TAGS = data.Field(unk_token = None)  # can't have unknown tags

        fields = (("text", TEXT), ("udtags", UD_TAGS), (None, None))

        train_data, valid_data, test_data = datasets.UDPOS.splits(fields, root='/home/pi/.data')

        TEXT.build_vocab(train_data, min_freq = MIN_FREQ)

        UD_TAGS.build_vocab(train_data)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        INPUT_DIM = len(TEXT.vocab)

        EMBEDDING_DIM = 100
        HIDDEN_DIM = 128

        OUTPUT_DIM = len(UD_TAGS.vocab)
        N_LAYERS = 2
        BIDIRECTIONAL = False
        DROPOUT = 0.25
        PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]


        self.model = BiLSTMPOSTagger(INPUT_DIM,
                        EMBEDDING_DIM,
                        HIDDEN_DIM,
                        OUTPUT_DIM,
                        N_LAYERS,
                        BIDIRECTIONAL,
                        DROPOUT,
                        PAD_IDX)

        self.model.apply(init_weights)
        # print(f'The model has {count_parameters(model):,} trainable parameters')
        self.model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)
        TAG_PAD_IDX = UD_TAGS.vocab.stoi[UD_TAGS.pad_token]

        self.optimizer = optim.Adam(self.model.parameters(),lr=LR)

    def apply_gradients(self, *gradients):
        summed_gradients = [
            np.stack(gradient_zip).sum(axis=0)
            for gradient_zip in zip(*gradients)
        ]
        self.optimizer.zero_grad()
        self.model.set_gradients(summed_gradients)
        self.optimizer.step()
        return self.model.get_weights()

    def get_weights(self):
        return self.model.get_weights()


@ray.remote(num_cpus=3)
class DataWorker(object):
    def __init__(self, rank):
        self.rank = rank

        TEXT = data.Field(lower = True)  # can have unknown tokens
        UD_TAGS = data.Field(unk_token = None)  # can't have unknown tags

        # don't load PTB tags
        fields = (("text", TEXT), ("udtags", UD_TAGS), (None, None))

        train_data, valid_data, test_data = datasets.UDPOS.splits(fields, root='/home/pi/.data')

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
        TAG_PAD_IDX = UD_TAGS.vocab.stoi[UD_TAGS.pad_token]

        self.model = BiLSTMPOSTagger(INPUT_DIM,
                        EMBEDDING_DIM,
                        HIDDEN_DIM,
                        OUTPUT_DIM,
                        N_LAYERS,
                        BIDIRECTIONAL,
                        DROPOUT,
                        PAD_IDX)

        self.data_iterator = iter(train_iterators[self.rank])
        self.criterion = nn.CrossEntropyLoss(ignore_index = TAG_PAD_IDX)

    def compute_gradients(self, weights):
        # print(f'computing gradients on node {self.rank} ...')
        self.model.set_weights(weights)

        try:
            batch = next(self.data_iterator)
        except StopIteration:  # When the epoch ends, start a new epoch.
            self.data_iterator = iter(train_iterators[self.rank])
            data, target = next(self.data_iterator)

        text = batch.text
        tags = batch.udtags
        # TODO: ?
        # optimizer.zero_grad()
        self.model.zero_grad()
        predictions = self.model(text)
        predictions = predictions.view(-1, predictions.shape[-1])
        tags = tags.view(-1)
        loss = self.criterion(predictions, tags)
        loss.backward()
        # print(f'finished computing gradients on node {self.rank}')
        return self.model.get_gradients()

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.normal_(param.data, mean = 0, std = 0.1)

def run():
    ray.init(
        address='auto',
        ignore_reinit_error=True,
        webui_host='0.0.0.0',
        redis_password='5241590000000000'
    )
    ps = ParameterServer.remote()
    workers = [DataWorker.remote(i) for i in range(PARALLELISM_LEVEL)]

    # print("Running synchronous parameter server training.")
    current_weights = ps.get_weights.remote()
    for i in range(5):
        gradients = [
            worker.compute_gradients.remote(current_weights) for worker in workers
        ]
        # print('gathering gradients...')
        current_weights = ps.apply_gradients.remote(*gradients)
        print(f'weights after batch {i}: {ray.get(current_weights).keys()}')

    # ray.shutdown()

if __name__ == "__main__":
    run()
