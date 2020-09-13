from datetime import datetime
import os
import random

import ray
import torch

from common.trainer import Trainer
from .ray_gru_pos_tagger_model import RayGRUPOSTaggerModel

MIN_FREQ = int(os.environ['SUSML_MIN_FREQ'])
RAND_SEED = int(os.environ['SUSML_RAND_SEED'])
NUM_EPOCHS = int(os.environ['SUSML_NUM_EPOCHS'])
BATCH_SIZE = int(os.environ['SUSML_BATCH_SIZE'])
LR = float(os.environ['SUSML_LR'])
EVAL_BETWEEN_BATCHES = True if os.environ['SUSML_EVAL_BETWEEN_BATCHES'] == 'true' else False
EVAL_EVERY_X_BATCHES = int(os.environ['SUSML_EVAL_EVERY_X_BATCHES'])
PARALLELISM_LEVEL = int(os.environ['SUSML_PARALLELISM_LEVEL'])

random.seed(RAND_SEED)
torch.manual_seed(RAND_SEED)
torch.backends.cudnn.deterministic = True


# won't really use 4 cpu cores because it's limited to 3 by OMP_NUM_THREADS=3
# @ray.remote(num_cpus=4)
@ray.remote(num_cpus=3)
class DataWorker(Trainer):
    def __init__(self, rank):
        super().__init__()

        self.rank = rank
        self.epoch_loss = 0
        self.epoch_acc = 0
        self.batch_idx = 1

        self.model = RayGRUPOSTaggerModel(self.INPUT_DIM,
                                          self.EMBEDDING_DIM,
                                          self.HIDDEN_DIM,
                                          self.OUTPUT_DIM,
                                          self.N_LAYERS,
                                          self.BIDIRECTIONAL,
                                          self.DROPOUT,
                                          self.PAD_IDX)

        self.data_iterator = iter(self.train_iterators[self.rank])

    def get_rank(self):
        return self.rank

    def compute_gradients(self, weights):
        self.model.set_weights(weights)

        try:
            batch = next(self.data_iterator)
            self.batch_idx += 1
        except StopIteration:  # When the epoch ends, start a new epoch.
            self.data_iterator = iter(self.train_iterators[self.rank])
            batch = next(self.data_iterator)
            self.batch_idx = 1

        text = batch.text
        tags = batch.udtags
        self.model.zero_grad()
        predictions = self.model(text)
        predictions = predictions.view(-1, predictions.shape[-1])
        tags = tags.view(-1)
        loss = self.criterion(predictions, tags)
        loss.backward()

        if EVAL_BETWEEN_BATCHES and (self.batch_idx % EVAL_EVERY_X_BATCHES == 0):
            valid_loss, valid_acc = self.evaluate(self.valid_iterator, 'between batches', -1, silent=True)
            print(f'Evaluating after batch {self.batch_idx} on rank {self.get_rank()}: Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')
            self.model.train()

        return self.model.get_gradients()
