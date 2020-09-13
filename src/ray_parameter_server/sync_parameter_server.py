import os
import random

import ray
import torch
import torch.optim as optim

from .abstract_parameter_server import AbstractParameterServer
from .data_worker import DataWorker


MIN_FREQ = int(os.environ['SUSML_MIN_FREQ'])
RAND_SEED = int(os.environ['SUSML_RAND_SEED'])
NUM_EPOCHS = int(os.environ['SUSML_NUM_EPOCHS'])
BATCH_SIZE = int(os.environ['SUSML_BATCH_SIZE'])
LR = float(os.environ['SUSML_LR'])
PARALLELISM_LEVEL = int(os.environ['SUSML_PARALLELISM_LEVEL'])

random.seed(RAND_SEED)
torch.manual_seed(RAND_SEED)
torch.backends.cudnn.deterministic = True


@ray.remote(num_cpus=1)
class SyncParameterServer(AbstractParameterServer):
    def __init__(self):
        super().__init__()

        self.workers = [DataWorker.remote(i) for i in range(PARALLELISM_LEVEL)]

        self.model.apply(self.init_weights)
        self.model.embedding.weight.data[self.PAD_IDX] = torch.zeros(self.EMBEDDING_DIM)

        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)

    def train(self):
        self.model.train()
        current_weights = self.get_weights()

        for _ in self.train_iterators[0]:
            gradients = [
                worker.compute_gradients.remote(current_weights) for worker in self.workers
            ]
            current_weights = self.apply_gradients(*ray.get(gradients))
