mkdir -p ../logs

TIMESTAMP="$(date +'%m-%d-%Y--%H-%M-%S')"
DIR_PATH="../logs/$TIMESTAMP"
mkdir $DIR_PATH

if [ "$OMPI_COMM_WORLD_RANK" = "0" ]
then
    cp ./config.ini "$DIR_PATH/config.ini"
    cp ./hostfile "$DIR_PATH/hostfile"
    cp ./train_eval_test.sh "$DIR_PATH/train_eval_test.sh"

    # # better but doesn't work syntax-wise somehow: 'https://stackoverflow.com/a/36625791
    # echo "BASH_SOURCE $BASH_SOURCE"
    # echo "@ $@"
    # echo "$BASH_SOURCE $@" >> "$DIR_PATH/invocation_command"
fi

# this works with `mpirun -n 2 -hostfile ./hostfile --mca orte_fork_agent sh ./train_eval_test.sh` as well as horovodrun
OMPI_MCA_opal_event_include=poll python3 -u ./src/lstm_ddp.py 2>&1 | tee "$DIR_PATH/$OMPI_COMM_WORLD_RANK.out"
