#!/usr/bin/env python
import os
import random

import torch

from horovod_adasum.horovod_adasum import HorovodAdaSum

MIN_FREQ = int(os.environ['SUSML_MIN_FREQ'])
RAND_SEED = int(os.environ['SUSML_RAND_SEED'])
NUM_EPOCHS = int(os.environ['SUSML_NUM_EPOCHS'])
BATCH_SIZE = int(os.environ['SUSML_BATCH_SIZE'])
LR = float(os.environ['SUSML_LR'])
PARALLELISM_LEVEL = int(os.environ['SUSML_PARALLELISM_LEVEL'])
RAY_PARAMETER_SERVER_STRATEGY = os.environ['RAY_PARAMETER_SERVER_STRATEGY']

random.seed(RAND_SEED)
torch.manual_seed(RAND_SEED)
torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    HorovodAdaSum().run()
