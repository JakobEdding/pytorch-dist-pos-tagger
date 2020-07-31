#!/bin/bash
source ./src/env.sh

# has only been created on master so far...
mkdir -p $SUSML_DIR_PATH

echo "this is hostname $(hostname)"
if [[ $(hostname) == pi* ]]
then
    echo "am on pi cluster"
    source ~/susml/jakob_jonas/bin/activate
fi

sar 1 > "$SUSML_DIR_PATH/$OMPI_COMM_WORLD_RANK-sar.out" &
SUSML_SAR_PID=$!

# MPI:
OMPI_MCA_opal_event_include=poll python3 -u ./src/mpi_ddp/rnn_mpi_ddp.py 2>&1 | tee "$SUSML_DIR_PATH/$OMPI_COMM_WORLD_RANK.out"

# Horovod:
# OMPI_MCA_opal_event_include=poll python3 -u ./src/horovod/rnn_horovod.py 2>&1 | tee "$SUSML_DIR_PATH/$OMPI_COMM_WORLD_RANK.out"

kill $SUSML_SAR_PID
unset "${!SUSML_@}"
