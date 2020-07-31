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

from data_worker import DataWorker

# preprocessing
MIN_FREQ = int(os.environ['SUSML_MIN_FREQ'])
# training
RAND_SEED = int(os.environ['SUSML_RAND_SEED'])
NUM_EPOCHS = int(os.environ['SUSML_NUM_EPOCHS'])
BATCH_SIZE = int(os.environ['SUSML_BATCH_SIZE'])
LR = float(os.environ['SUSML_LR'])
# distribution
PARALLELISM_LEVEL = int(os.environ['SUSML_PARALLELISM_LEVEL'])
# print('parallelism level is', PARALLELISM_LEVEL)

random.seed(RAND_SEED)
torch.manual_seed(RAND_SEED)
torch.backends.cudnn.deterministic = True

@ray.remote(num_cpus=1)
class ParameterServer(object):
    def __init__(self):
        self.workers = [DataWorker.remote(i) for i in range(PARALLELISM_LEVEL)]

        TEXT = data.Field(lower = True)  # can have unknown tokens
        UD_TAGS = data.Field(unk_token = None)  # can't have unknown tags

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

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.train_iterators = data.BucketIterator.splits(
            train_data_tuple,
            batch_size = BATCH_SIZE,
            device = device)

        self.valid_iterator, self.test_iterator = data.BucketIterator.splits(
            (valid_data, test_data),
            batch_size = BATCH_SIZE,
            device = device)

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

        self.model.apply(self.init_weights)
        # print(f'The model has {count_parameters(model):,} trainable parameters')
        self.model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)
        self.TAG_PAD_IDX = UD_TAGS.vocab.stoi[UD_TAGS.pad_token]

        self.criterion = nn.CrossEntropyLoss(ignore_index = self.TAG_PAD_IDX)

        self.optimizer = optim.Adam(self.model.parameters(),lr=LR)

    def epoch_time(self, start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs

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

    def init_weights(self, m):
        for name, param in m.named_parameters():
            nn.init.normal_(param.data, mean = 0, std = 0.1)

    def train(self):
        # model, train_iterators[0], optimizer, criterion, TAG_PAD_IDX, rank, epoch
        # epoch_loss = 0
        # epoch_acc = 0

        self.model.train()

        current_weights = self.get_weights()

        for batch_idx, batch in enumerate(self.train_iterators[0]):
            # import pdb;pdb.set_trace()
            # print('beginning new batch')
            gradients = [
                worker.compute_gradients.remote(current_weights) for worker in self.workers
            ]
            # print('gathering gradients...')
            current_weights = self.apply_gradients(*ray.get(gradients))
            # print(f'weights after batch {i}: {ray.get(current_weights).keys()}')

        # return epoch_loss / len(iterator), epoch_acc / len(iterator)

    def categorical_accuracy(self, preds, y):
        """
        Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
        """
        max_preds = preds.argmax(dim = 1, keepdim = True) # get the index of the max probability
        non_pad_elements = (y != self.TAG_PAD_IDX).nonzero()
        correct = max_preds[non_pad_elements].squeeze(1).eq(y[non_pad_elements])
        return correct.sum() / torch.FloatTensor([y[non_pad_elements].shape[0]])


    def evaluate(self):

        # tag_pad_idx

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

        return epoch_loss / len(self.valid_iterator), epoch_acc / len(self.valid_iterator)


    def run(self):
        overall_start_time = time.time()

        for epoch in range(NUM_EPOCHS):
            print(f'Starting epoch {epoch+1:02}')
            start_time = time.time()
            # train_loss, train_acc = train()
            self.train()
            valid_loss, valid_acc = self.evaluate()
            end_time = time.time()
            epoch_mins, epoch_secs = self.epoch_time(start_time, end_time)
            print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            # print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

        overall_end_time = time.time()
        print('took overall', self.epoch_time(overall_start_time, overall_end_time))

        return 1

    def run_async(self):
        overall_start_time = time.time()

        current_weights = self.get_weights()


        updates = len(self.train_iterators[0]) * len(self.workers)
        for epoch in range(NUM_EPOCHS):
            gradients = {}
            for worker in self.workers:
                gradients[worker.compute_gradients.remote(current_weights)] = worker

            batches_processed_by_worker = {worker_id: 0 for worker_id in range(PARALLELISM_LEVEL)}
            start_time = time.time()

            for iteration in range(updates):
                print(f'Starting update {iteration+1:03}/{updates}')
                # train_loss, train_acc = train()
                ready_gradient_list, rest = ray.wait(list(gradients))
                if len(ready_gradient_list) == 0:
                    print(f'wait failed {ready_gradient_list}, {rest}')
                ready_gradient_id = ready_gradient_list[0]
                worker = gradients.pop(ready_gradient_id)
                worker_rank = ray.get(worker.get_rank.remote())
                batches_processed_by_worker[worker_rank] += 1
                self.model.train()
                current_weights = self.apply_gradients(*[ray.get(ready_gradient_id)])

                if batches_processed_by_worker[worker_rank] <= len(self.train_iterators[0]):
                    gradients[worker.compute_gradients.remote(current_weights)] = worker

                # print(f'Update: {iteration+1:02} | Update Time: {epoch_mins}m {epoch_secs}s')

            end_time = time.time()
            epoch_mins, epoch_secs = self.epoch_time(start_time, end_time)

            valid_loss, valid_acc = self.evaluate()
            # print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
            print(f'Finished epoch {epoch+1:02}')
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

        overall_end_time = time.time()
        valid_loss, valid_acc = self.evaluate()
        print(f'Final Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')
        print('took overall', self.epoch_time(overall_start_time, overall_end_time))

        return 1
