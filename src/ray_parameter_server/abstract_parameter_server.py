import os
import random
import time

import numpy as np
import ray
import torch
import torch.optim as optim

from common.trainer import Trainer
from .data_worker import DataWorker
from .ray_gru_pos_tagger_model import RayGRUPOSTaggerModel

MIN_FREQ = int(os.environ['SUSML_MIN_FREQ'])
RAND_SEED = int(os.environ['SUSML_RAND_SEED'])
NUM_EPOCHS = int(os.environ['SUSML_NUM_EPOCHS'])
BATCH_SIZE = int(os.environ['SUSML_BATCH_SIZE'])
LR = float(os.environ['SUSML_LR'])
PARALLELISM_LEVEL = int(os.environ['SUSML_PARALLELISM_LEVEL'])

random.seed(RAND_SEED)
torch.manual_seed(RAND_SEED)
torch.backends.cudnn.deterministic = True


class AbstractParameterServer(Trainer):
    def __init__(self):
        super().__init__()

        self.workers = [DataWorker.remote(i) for i in range(PARALLELISM_LEVEL)]

        self.model = RayGRUPOSTaggerModel(self.INPUT_DIM,
                                          self.EMBEDDING_DIM,
                                          self.HIDDEN_DIM,
                                          self.OUTPUT_DIM,
                                          self.N_LAYERS,
                                          self.BIDIRECTIONAL,
                                          self.DROPOUT,
                                          self.PAD_IDX)

        self.model.apply(self.init_weights)
        self.model.embedding.weight.data[self.PAD_IDX] = torch.zeros(self.EMBEDDING_DIM)

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
        print(f'Took overall: {total_mins}m {total_secs}s')

        return 1
