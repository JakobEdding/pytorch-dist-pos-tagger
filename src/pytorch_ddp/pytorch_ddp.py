#!/usr/bin/env python
import os
import time
import random

import torch
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

from common.trainer import Trainer

MIN_FREQ = int(os.environ['SUSML_MIN_FREQ'])
RAND_SEED = int(os.environ['SUSML_RAND_SEED'])
NUM_EPOCHS = int(os.environ['SUSML_NUM_EPOCHS'])
BATCH_SIZE = int(os.environ['SUSML_BATCH_SIZE'])
LR = float(os.environ['SUSML_LR'])
PARALLELISM_LEVEL = int(os.environ['SUSML_PARALLELISM_LEVEL'])

random.seed(RAND_SEED)
torch.manual_seed(RAND_SEED)
torch.backends.cudnn.deterministic = True


class PyTorchDDP(Trainer):
    def run(self, rank):
        self.model.apply(self.init_weights)

        self.model.embedding.weight.data[self.PAD_IDX] = torch.zeros(self.EMBEDDING_DIM)

        self.model.to(self.device)

        print('Rank ', rank, ' initial_model: ', sum(parameter.sum() for parameter in self.model.parameters()))
        self.model = DDP(self.model)
        print('Rank ', rank, ' initial_model: ', sum(parameter.sum() for parameter in self.model.parameters()))
        self.criterion = self.criterion.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)

        self.model.train()
        total_start_time = time.time()

        self.iterate_epochs(rank)

        self.test()

        total_end_time = time.time()
        total_mins, total_secs = self.diff_time(total_start_time, total_end_time)
        print(f'Took overall: {total_mins}m {total_secs}s')
