#!/bin/bash
source ./src/env.sh

# has only been created on master so far...
mkdir -p $SUSML_DIR_PATH

if [[ $(hostname) == pi* ]]
then
    echo "am on pi cluster"
    source ~/susml/jakob_jonas/bin/activate
fi
OMPI_MCA_opal_event_include=poll python3 -u ./src/horovod/rnn_horovod.py 2>&1 | tee "$SUSML_DIR_PATH/$OMPI_COMM_WORLD_RANK.out"

unset "${!SUSML_@}"
