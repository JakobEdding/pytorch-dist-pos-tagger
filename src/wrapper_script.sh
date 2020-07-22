SUSML_TIMESTAMP="$(date +'%m-%d-%Y--%H-%M-%S')"
# create log directory outside git repo in case git repo is sshfs-mounted to avoid networking/IO
SUSML_DIR_PATH="../../logs/$SUSML_TIMESTAMP"
mkdir -p $SUSML_DIR_PATH

# pipe current commit hash and diff into file for reproducibility
git rev-parse HEAD > "$SUSML_DIR_PATH/git_commit_hash.txt"
# make sure untracked files show up in the diff as well / mark them as tracked without staging them https://stackoverflow.com/a/857696
git add -N *
git status > "$SUSML_DIR_PATH/git_status.txt"
git diff > "$SUSML_DIR_PATH/git_diff.txt"
git diff --staged > "$SUSML_DIR_PATH/git_diff_staged.txt"

source ./env.sh

horovodrun --hostfile hostfile -np $SUSML_PARALLELISM_LEVEL --mpi-args="--map-by socket:pe=3 -x SUSML_DIR_PATH=$SUSML_DIR_PATH" bash ./train_eval_test.sh
#mpirun -x SUSML_DIR_PATH=$SUSML_DIR_PATH -n $SUSML_PARALLELISM_LEVEL --map-by socket:pe=3 -hostfile ./hostfile --mca orte_fork_agent bash ./train_eval_test.sh
# horovodrun

unset "${!SUSML_@}"
