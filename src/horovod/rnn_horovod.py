#!/usr/bin/env python
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
import horovod.torch as hvd
from torchtext import data
from torchtext import datasets
from tqdm import tqdm


from common.model.gru_pos_tagger_model import GRUPOSTaggerModel

# preprocessing
MIN_FREQ = int(os.environ['SUSML_MIN_FREQ'])
# training
RAND_SEED = int(os.environ['SUSML_RAND_SEED'])
NUM_EPOCHS = int(os.environ['SUSML_NUM_EPOCHS'])
BATCH_SIZE = int(os.environ['SUSML_BATCH_SIZE'])
LR = float(os.environ['SUSML_LR'])
# distribution
PARALLELISM_LEVEL = int(os.environ['SUSML_PARALLELISM_LEVEL'])

random.seed(RAND_SEED)
torch.manual_seed(RAND_SEED)
torch.backends.cudnn.deterministic = True

from common.distributed_trainer import DistributedTrainer

class RNNHorovod(DistributedTrainer):
    def run(self):
        hvd.init()

        # create local model
        local_model = GRUPOSTaggerModel(self.INPUT_DIM,
                            self.EMBEDDING_DIM,
                            self.HIDDEN_DIM,
                            self.OUTPUT_DIM,
                            self.N_LAYERS,
                            self.BIDIRECTIONAL,
                            self.DROPOUT,
                            self.PAD_IDX)

        local_model.apply(self.init_weights)
        rank = hvd.rank()
        # print(f'The model has {count_parameters(local_model):,} trainable parameters')
        local_model.embedding.weight.data[self.PAD_IDX] = torch.zeros(self.EMBEDDING_DIM)

        hvd.broadcast_parameters(local_model.state_dict(), root_rank=0)

        print('rank ', rank, ' initial_model: ', sum(parameter.sum() for parameter in local_model.parameters()))
        # construct DDP model
        #ddp_model = DDP(local_model)
        model = local_model
        print('rank ', rank, ' initial_ddp_model: ', sum(parameter.sum() for parameter in model.parameters()))
        # define loss function and optimizer
        criterion = nn.CrossEntropyLoss(ignore_index=self.TAG_PAD_IDX)
        # criterion = criterion.to(device)
        optimizer = optim.Adam(model.parameters(), lr=LR)

        # optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters(), op=hvd.Average)
        optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters(), op=hvd.Adasum)

        best_valid_loss = float('inf')
        total_start_time = time.time()
        #ddp_model.train()

        for epoch in range(NUM_EPOCHS):
            print(f'Starting epoch {epoch+1:02}')
            epoch_start_time = time.time()

            train_loss, train_acc = self.train(model, self.train_iterators[rank], optimizer, criterion, self.TAG_PAD_IDX, rank, epoch)
            valid_loss, valid_acc = self.evaluate(model, self.valid_iterator, criterion, self.TAG_PAD_IDX, 'valid', epoch)

            epoch_end_time = time.time()
            epoch_mins, epoch_secs = self.diff_time(epoch_start_time, epoch_end_time)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                if not os.path.exists('../tmp_model'):
                    os.makedirs('../tmp_model')
                # save model outside directory that could be sshfs-mounted
                torch.save(model.state_dict(), '../tmp_model/model.pt')

            print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

        model.load_state_dict(torch.load('../tmp_model/tut1-model.pt'))
        test_loss, test_acc = self.evaluate(model, self.test_iterator, criterion, self.TAG_PAD_IDX, 'test', -1)
        print(f'Test Loss: {test_loss:.3f} |  Test Acc: {test_acc*100:.2f}%')

        total_end_time = time.time()
        total_mins, total_secs = self.diff_time(total_start_time, total_end_time)
        print(f'took overall: {total_mins}m {total_secs}s')

if __name__ == "__main__":
    RNNHorovod().run()
