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

from parameter_server import ParameterServer

# preprocessing
MIN_FREQ = int(os.environ['SUSML_MIN_FREQ'])
# training
RAND_SEED = int(os.environ['SUSML_RAND_SEED'])
NUM_EPOCHS = int(os.environ['SUSML_NUM_EPOCHS'])
BATCH_SIZE = int(os.environ['SUSML_BATCH_SIZE'])
LR = float(os.environ['SUSML_LR'])
# model
RNN_LAYER_TYPE = os.environ['SUSML_RNN_LAYER_TYPE']
# distribution
PARALLELISM_LEVEL = int(os.environ['SUSML_PARALLELISM_LEVEL'])
print('parallelism level is', PARALLELISM_LEVEL)

random.seed(RAND_SEED)
torch.manual_seed(RAND_SEED)
torch.backends.cudnn.deterministic = True

def run():
    ray.init(
        address='auto',
        ignore_reinit_error=True,
        webui_host='0.0.0.0',
        redis_password='5241590000000000'
    )
    try:
        ps = ParameterServer.remote()
        val = ps.run_async.remote()
        print(ray.get(val))
    except Exception as e:
        raise e
    finally:
        print('waiting 10s to allow logs to flush')
        time.sleep(10)
        ray.shutdown()


if __name__ == "__main__":
    run()
