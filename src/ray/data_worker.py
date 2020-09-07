#!/usr/bin/env python
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

import ray

import numpy as np

from ray_gru_pos_tagger_model import RayGRUPOSTaggerModel

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

from common.distributed_trainer import DistributedTrainer

# won't really use 4 cpu cores because it's limited to 3 by OMP_NUM_THREADS=3
# @ray.remote(num_cpus=4)
@ray.remote(num_cpus=3)
class DataWorker(DistributedTrainer):
    def __init__(self, rank):
        self.rank = rank
        self.epoch_loss = 0
        self.epoch_acc = 0
        self.batch_idx = 1


        self.model = RayGRUPOSTaggerModel(INPUT_DIM,
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
        sequence_lengths = []
        for data_example in text:
            # remove padding
            sequence_lengths.append(len(list(filter(lambda x: x != 1, data_example))))
        summed_length_of_sequences = sum(sequence_lengths)
        print(f'computed gradients for a batch on node {self.rank}, took {(later-before).seconds:03}.{(later-before).microseconds}, summed length of batch sequences: {summed_length_of_sequences}, individual lengths: {sequence_lengths}')

        return self.model.get_gradients()
