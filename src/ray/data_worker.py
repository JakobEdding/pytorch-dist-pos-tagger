#!/usr/bin/env python
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

import ray

import numpy as np

from gru_pos_tagger import GRUPOSTagger

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
EVAL_BETWEEN_BATCHES = True if os.environ['SUSML_EVAL_BETWEEN_BATCHES'] == 'true' else False
EVAL_EVERY_X_BATCHES = int(os.environ['SUSML_EVAL_EVERY_X_BATCHES'])
# distribution
PARALLELISM_LEVEL = int(os.environ['SUSML_PARALLELISM_LEVEL'])
# print('parallelism level is', PARALLELISM_LEVEL)

print(f'RAND-TEST rand seed in data_worker.py is {RAND_SEED}')
random.seed(RAND_SEED)
torch.manual_seed(RAND_SEED)
torch.backends.cudnn.deterministic = True

# won't really use 4 cpu cores because it's limited to 3 by OMP_NUM_THREADS=3
# @ray.remote(num_cpus=4)
@ray.remote(num_cpus=3)
class DataWorker(object):
    def __init__(self, rank):
        self.rank = rank
        self.epoch_loss = 0
        self.epoch_acc = 0
        self.batch_idx = 1

        TEXT = data.Field(lower = True)  # can have unknown tokens
        UD_TAGS = data.Field(unk_token = None)  # can't have unknown tags

        # don't load PTB tags
        fields = (("text", TEXT), ("udtags", UD_TAGS), (None, None))

        train_data, valid_data, test_data = datasets.UDPOS.splits(fields, root='/home/pi/.data')

        # inspired by torchtext internals because their splits method is limited to 3 workers... https://github.com/pytorch/text/blob/e70955309ead681f924fecd36d759c37e3fdb1ee/torchtext/data/dataset.py#L325
        def custom_split(examples, number_of_parts):
            N = len(examples)
            randperm = random.sample(range(N), len(range(N)))
            print(f'RAND-TEST first three elements of randperm in data_worker.py for rank {self.rank} are {randperm[:3]}')
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

        self.train_iterators = data.BucketIterator.splits(
            train_data_tuple,
            batch_size = BATCH_SIZE,
            device = device)

        self.valid_iterator, test_iterator = data.BucketIterator.splits(
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
        self.TAG_PAD_IDX = UD_TAGS.vocab.stoi[UD_TAGS.pad_token]

        self.model = GRUPOSTagger(INPUT_DIM,
                        EMBEDDING_DIM,
                        HIDDEN_DIM,
                        OUTPUT_DIM,
                        N_LAYERS,
                        BIDIRECTIONAL,
                        DROPOUT,
                        PAD_IDX)

        self.data_iterator = iter(self.train_iterators[self.rank])
        self.criterion = nn.CrossEntropyLoss(ignore_index = self.TAG_PAD_IDX)

    # def clear_epoch_metrics():
    #     self.epoch_loss = 0
    #     self.epoch_acc = 0

    def get_rank(self):
        return self.rank

    def categorical_accuracy(self, preds, y):
        """
        Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
        """
        max_preds = preds.argmax(dim = 1, keepdim = True) # get the index of the max probability
        non_pad_elements = (y != self.TAG_PAD_IDX).nonzero()
        correct = max_preds[non_pad_elements].squeeze(1).eq(y[non_pad_elements])
        return correct.sum() / torch.FloatTensor([y[non_pad_elements].shape[0]])

    def compute_gradients(self, weights):
        # print(f'computing gradients for a batch on node {self.rank} at {datetime.now()}...')
        self.model.set_weights(weights)

        try:
            batch = next(self.data_iterator)
            self.batch_idx += 1
        except StopIteration:  # When the epoch ends, start a new epoch.
            # print(f'starting new epoch on rank {self.get_rank()}')
            self.data_iterator = iter(self.train_iterators[self.rank])
            batch = next(self.data_iterator)
            self.batch_idx = 1

        before = datetime.now()
        text = batch.text
        tags = batch.udtags
        # TODO: ?
        # optimizer.zero_grad()
        self.model.zero_grad()
        predictions = self.model(text)
        predictions = predictions.view(-1, predictions.shape[-1])
        tags = tags.view(-1)
        loss = self.criterion(predictions, tags)
        # self.epoch_acc += categorical_accuracy(predictions, tags, tag_pad_idx).item()
        # acc = categorical_accuracy(predictions, tags, tag_pad_idx)
        loss.backward()
        # self.epoch_loss += loss.item()

        # TODO: assert hashes before and after evaluating of model.get_gradients() are the same!

        if EVAL_BETWEEN_BATCHES and (self.batch_idx % EVAL_EVERY_X_BATCHES == 0):
            epoch_loss = 0
            epoch_acc = 0

            # TODO: set model.train() somewhere else before applying workers' gradients!?
            self.model.eval()

            with torch.no_grad():
                for batch in self.valid_iterator:
                    text = batch.text
                    tags = batch.udtags
                    predictions = self.model(text)
                    predictions = predictions.view(-1, predictions.shape[-1])
                    tags = tags.view(-1)
                    loss = self.criterion(predictions, tags)
                    acc = self.categorical_accuracy(predictions, tags)
                    epoch_loss += loss.item()
                    epoch_acc += acc.item()

            valid_loss, valid_acc = (epoch_loss / len(self.valid_iterator), epoch_acc / len(self.valid_iterator))
            print(f'evaluating after batch {self.batch_idx} on rank {self.get_rank()}: Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')
            self.model.train()

        later = datetime.now()
        print(f'computed gradients for a batch on node {self.rank}, took {(later-before).seconds:03}.{(later-before).microseconds}, batch is: {text}')

        return self.model.get_gradients()
