#!/usr/bin/env python
import os
import time
import random

import horovod.torch as hvd
import torch
import torch.optim as optim

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


class HorovodAdaSum(Trainer):
    def run(self):
        hvd.init()

        self.model.apply(self.init_weights)
        rank = hvd.rank()
        self.model.embedding.weight.data[self.PAD_IDX] = torch.zeros(self.EMBEDDING_DIM)

        hvd.broadcast_parameters(self.model.state_dict(), root_rank=0)

        print('Rank ', rank, ' initial_model: ', sum(parameter.sum() for parameter in self.model.parameters()))
        print('Rank ', rank, ' initial_ddp_model: ', sum(parameter.sum() for parameter in self.model.parameters()))
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)

        # self.optimizer = hvd.DistributedOptimizer(self.optimizer, named_parameters=model.named_parameters(), op=hvd.Average)
        self.optimizer = hvd.DistributedOptimizer(self.optimizer, named_parameters=self.model.named_parameters(), op=hvd.Adasum)

        total_start_time = time.time()

        self.iterate_epochs(rank)

        self.test()

        total_end_time = time.time()
        total_mins, total_secs = self.diff_time(total_start_time, total_end_time)
        print(f'Took overall: {total_mins}m {total_secs}s')
