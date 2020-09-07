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

print(f'RAND-TEST rand seed in parameter_server.py is {RAND_SEED}')
random.seed(RAND_SEED)
torch.manual_seed(RAND_SEED)
torch.backends.cudnn.deterministic = True

from common.distributed_trainer import DistributedTrainer

@ray.remote(num_cpus=1)
class ParameterServer(DistributedTrainer):
    def __init__(self):
        self.workers = [DataWorker.remote(i) for i in range(PARALLELISM_LEVEL)]

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = RayGRUPOSTaggerModel(self.INPUT_DIM,
                        self.EMBEDDING_DIM,
                        self.HIDDEN_DIM,
                        self.OUTPUT_DIM,
                        self.N_LAYERS,
                        self.BIDIRECTIONAL,
                        self.DROPOUT,
                        self.PAD_IDX)

        self.model.apply(self.init_weights)
        print(f'RAND-TEST hash of random-initialized model weights in parameter_server.py {hash(str(self.model.get_weights()))}')
        self.model.embedding.weight.data[self.PAD_IDX] = torch.zeros(self.EMBEDDING_DIM)

        self.criterion = nn.CrossEntropyLoss(ignore_index=self.TAG_PAD_IDX)

        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)

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

    def train(self):
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

    def evaluate(self, iterator, method, epoch):
        epoch_loss = 0
        epoch_acc = 0
        start_time = time.time()

        # TODO: set model.train() somewhere else before applying workers' gradients!?
        self.model.eval()

        with torch.no_grad():
            for batch in iterator:
                text = batch.text
                tags = batch.udtags
                predictions = self.model(text)
                predictions = predictions.view(-1, predictions.shape[-1])
                tags = tags.view(-1)
                loss = self.criterion(predictions, tags)
                acc = self.categorical_accuracy(predictions, tags)
                epoch_loss += loss.item()
                epoch_acc += acc.item()

        end_time = time.time()
        mins, secs = self.diff_time(start_time, end_time)
        print(f'Epoch {epoch+1:02} {method} time: {mins}m {secs}s')

        return epoch_loss / len(iterator), epoch_acc / len(iterator)

    def run(self):
        total_start_time = time.time()

        for epoch in range(NUM_EPOCHS):
            print(f'Starting epoch {epoch+1:02}')
            epoch_start_time = time.time()

            train_start_time = time.time()
            self.train()
            train_end_time = time.time()
            train_mins, train_secs = self.diff_time(train_start_time, train_end_time)
            print(f'Epoch {epoch+1:02} train time: {train_mins}m {train_secs}s')

            valid_loss, valid_acc = self.evaluate(self.valid_iterator, 'valid', epoch)

            epoch_end_time = time.time()
            epoch_mins, epoch_secs = self.diff_time(epoch_start_time, epoch_end_time)

            print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

        test_loss, test_acc = self.evaluate(self.test_iterator, 'test', -1)
        print(f'Test Loss: {test_loss:.3f} |  Test Acc: {test_acc*100:.2f}%')

        total_end_time = time.time()
        total_mins, total_secs = self.diff_time(total_start_time, total_end_time)
        print(f'took overall: {total_mins}m {total_secs}s')

        return 1

    def run_async(self):
        total_start_time = time.time()
        current_weights = self.get_weights()

        updates = len(self.train_iterators[0]) * len(self.workers)
        for epoch in range(NUM_EPOCHS):
            print(f'Starting epoch {epoch+1:02}')
            epoch_start_time = time.time()

            train_start_time = time.time()
            gradients = {}
            for worker in self.workers:
                gradients[worker.compute_gradients.remote(current_weights)] = worker

            batches_processed_by_worker = {worker_id: 0 for worker_id in range(PARALLELISM_LEVEL)}

            for iteration in range(updates):
                # print(f'Starting update {iteration+1:03}/{updates}')
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

            train_end_time = time.time()
            train_mins, train_secs = self.diff_time(train_start_time, train_end_time)
            print(f'Epoch {epoch+1:02} train time: {train_mins}m {train_secs}s')

            valid_loss, valid_acc = self.evaluate(self.valid_iterator, 'valid', epoch)

            epoch_end_time = time.time()
            epoch_mins, epoch_secs = self.diff_time(epoch_start_time, epoch_end_time)

            print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

        test_loss, test_acc = self.evaluate(self.test_iterator, 'test', -1)
        print(f'Test Loss: {test_loss:.3f} |  Test Acc: {test_acc*100:.2f}%')

        total_end_time = time.time()
        total_mins, total_secs = self.diff_time(total_start_time, total_end_time)
        print(f'took overall: {total_mins}m {total_secs}s')

        return 1
