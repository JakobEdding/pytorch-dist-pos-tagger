#!/bin/bash

# Distribution
export SUSML_PARALLELISM_LEVEL=12
# export RAY_PARAMETER_SERVER_STRATEGY='sync'  # for synchronous parameter server architecture
export RAY_PARAMETER_SERVER_STRATEGY='async'  # for asynchronous parameter server architecture

# Preprocessing
export SUSML_MIN_FREQ=2

# Training
export SUSML_RAND_SEED=1234
export SUSML_NUM_EPOCHS=5
export SUSML_BATCH_SIZE=128
export SUSML_LR=0.001
# Only applies to Ray parameter server approaches
export SUSML_EVAL_BETWEEN_BATCHES=false
export SUSML_EVAL_EVERY_X_BATCHES=2
