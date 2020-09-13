#!/usr/bin/env python
import os
import time
import random

import ray
import torch

from ray_parameter_server.async_parameter_server import AsyncParameterServer
from ray_parameter_server.sync_parameter_server import SyncParameterServer

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


def start():
    ray.init(
        # address='192.168.178.51:6379',
        address='auto',
        ignore_reinit_error=True,
        webui_host='0.0.0.0',
        redis_password='5241590000000000'
    )
    try:
        if RAY_PARAMETER_SERVER_STRATEGY == 'sync':
            parameter_server = SyncParameterServer.remote()
        elif RAY_PARAMETER_SERVER_STRATEGY == 'async':
            parameter_server = AsyncParameterServer.remote()
        else:
            raise Exception('Environment variable "RAY_PARAMETER_SERVER_STRATEGY" must be set in ./src/env.sh to either "sync" or "async" to use one of the parameter server architectures.')
        print(ray.get(parameter_server.run.remote()))
    except Exception as e:
        raise e
    finally:
        print('Waiting 10s to allow logs to flush')
        time.sleep(10)
        ray.shutdown()


if __name__ == "__main__":
    start()
