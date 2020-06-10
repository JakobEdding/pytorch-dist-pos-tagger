cp ./config.ini "./logs/config-$(date +'%m-%d-%Y, %H-%M-%S').ini"
echo "config.ini"
cat ./config.ini
cp ./hostfile "./logs/hostfile-$(date +'%m-%d-%Y, %H-%M-%S')"
echo "hostfile"
cat hostfile

echo "RNN_TYPE is $RNN_TYPE"

# doesn't log stderr so far
# this kind of works at least when executed like this `mpirun -n 2 -hostfile ./hostfile --mca orte_fork_agent sh ./train_eval_test.sh`
# OMPI_MCA_opal_event_include=poll python3 -u ./src/lstm_ddp.py > ./logs/out-$OMPI_COMM_WORLD_RANK-$(date +"%m-%d-%Y, %H-%M-%S")
# this works well with tqdm(..., file=sys.stdout)
# OMPI_MCA_opal_event_include=poll python3 -u ./src/lstm_ddp.py | tee "./logs/out-$OMPI_COMM_WORLD_RANK-$(date +'%m-%d-%Y, %H-%M-%S')"

# this works with `mpirun -n 2 -hostfile ./hostfile --mca orte_fork_agent sh ./train_eval_test.sh`
OMPI_MCA_opal_event_include=poll python3 -u ./src/lstm_ddp.py 2>&1 | tee "../logs/out-$OMPI_COMM_WORLD_RANK-$(date +'%m-%d-%Y, %H-%M-%S')"
