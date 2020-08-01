SUSML_TIMESTAMP="$(date +'%Y-%m-%d--%H-%M-%S')"
# create log directory outside git repo in case git repo is sshfs-mounted to avoid networking/IO
SUSML_DIR_PATH="../logs/$SUSML_TIMESTAMP"
mkdir -p $SUSML_DIR_PATH

# pipe current commit hash and diff into file for reproducibility
git rev-parse HEAD > "$SUSML_DIR_PATH/git_commit_hash.txt"
# make sure untracked files show up in the diff as well / mark them as tracked without staging them https://stackoverflow.com/a/857696
git add -N *
git status > "$SUSML_DIR_PATH/git_status.txt"
git diff > "$SUSML_DIR_PATH/git_diff.txt"
git diff --staged > "$SUSML_DIR_PATH/git_diff_staged.txt"

source ./src/env.sh

# MPI:
# mpirun -x SUSML_DIR_PATH=$SUSML_DIR_PATH -n $SUSML_PARALLELISM_LEVEL --map-by socket:pe=3 -hostfile ./hostfile --mca orte_fork_agent bash ./src/train_eval_test.sh


# HOROVOD:
# if [[ $(hostname) == pi* ]]
# then
#     echo "am on pi cluster"
#     source ~/susml/jakob_jonas/bin/activate
# fi
# horovodrun --hostfile ./hostfile -np $SUSML_PARALLELISM_LEVEL --mpi-args="--map-by socket:pe=3 -x SUSML_DIR_PATH=$SUSML_DIR_PATH" bash ./src/train_eval_test.sh


# RAY:

# MASTER:
# source ~/susml/jakob_jonas/bin/activate
# ray stop && source ./src/env.sh && OMP_NUM_THREADS=3 ray start --head --port=6379 --webui-host='0.0.0.0'
# sar 1 > "ray-sar.out" &
# SAR_PID=$!

# SLAVES: (TODO: change master address)
# source ~/susml/jakob_jonas/bin/activate
# ray stop && source ./src/env.sh && OMP_NUM_THREADS=3 ray start --address='10.42.0.50:6379' --redis-password='5241590000000000'
# sar 1 > "ray-sar.out" &
# SAR_PID=$!

# START:
python3 -u src/ray/start.py 2>&1 | tee "$SUSML_DIR_PATH/0.out"

# STOP:
# kill $SAR_PID
# TODO: manually copy ray-sar.out into correct log directory

unset "${!SUSML_@}"
