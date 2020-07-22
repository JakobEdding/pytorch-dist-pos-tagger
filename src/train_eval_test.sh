#!/bin/bash
source ./env.sh

# echo "test"
# echo "$SUSML_DIR_PATH"
mkdir -p $SUSML_DIR_PATH # has only been created on master so far...

source ~/susml/jakob_jonas/bin/activate && OMPI_MCA_opal_event_include=poll python3 -u ./src/rnn_horovod.py 2>&1 | tee "$SUSML_DIR_PATH/$OMPI_COMM_WORLD_RANK.out"

unset "${!SUSML_@}"
