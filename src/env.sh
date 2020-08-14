#!/bin/bash

# preprocessing
export SUSML_MIN_FREQ=2

# training
export SUSML_RAND_SEED=1234
export SUSML_NUM_EPOCHS=5
export SUSML_BATCH_SIZE=128
export SUSML_LR=0.001
#  linear scaling rule for batch_size <-> lr ...
# export SUSML_BATCH_SIZE=512
# export SUSML_LR=0.004
export SUSML_EVAL_BETWEEN_BATCHES=false
export SUSML_EVAL_EVERY_X_BATCHES=2

# distribution / mpi / horovod
export SUSML_PARALLELISM_LEVEL=2
