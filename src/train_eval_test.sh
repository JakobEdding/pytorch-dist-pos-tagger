#!/bin/bash
source ./env.sh

# has only been created on master so far...
mkdir -p $SUSML_DIR_PATH

source ~/susml/jakob_jonas/bin/activate && OMPI_MCA_opal_event_include=poll python3 -u ./src/horovod/rnn_horovod.py 2>&1 | tee "$SUSML_DIR_PATH/$OMPI_COMM_WORLD_RANK.out"

unset "${!SUSML_@}"
