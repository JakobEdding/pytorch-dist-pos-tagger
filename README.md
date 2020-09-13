# Part-of-Speech tagging using RNNs on low-spec hardware clusters

This repository contains the code to reproduce the results from Section 4 of "Sustainable Machine Learning On Edge Device Clusters".

## Setup

1. Install Python3 and Virtualenv on all nodes of the cluster:
```
sudo apt-get install python3.6 virtualenv
```
2. Install OpenMPI on all nodes (see [here](https://www.open-mpi.org/))
3. Create an environment and install all requirements, some may need to be built specifically for the ARM platform, then distribute it to all nodes:
```
# Install Horovod manually with only MPI for backend and PyTorch for deep learning framework support:
HOROVOD_WITHOUT_GLOO=1 HOROVOD_WITH_MPI=1 HOROVOD_WITH_PYTORCH=1 pip install horovod[pytorch]==0.19.4
pip install -r requirements.txt
```
4. Make sure, this repository is available on all nodes.

## Running approaches an a Raspberry PI cluster:

1. Adapt `./hostfile` to contain all nodes that should partake in training.
2. Adapt `./src/env.sh` to reflect desired training parameters, specifically set parallelism to number of nodes in cluster.
3. Uncomment relevant section for approach in `./src/wrapper_script.sh`. Make sure, path to environment matches your local settings.
4. Next:
    - **For Ray:** Follow instructions denoted by `MASTER` and `SLAVES` in wrapper script to start the ray cluster, i.e. start head process on head node and join this processes ray cluster on all other nodes. (Do this before starting the wrapper script!). Plus, choose asynchronous or synchronous approach by setting the variable `RAY_PARAMETER_SERVER_STRATEGY` in `./src/env.sh` appropriately. In theory, it should be possible to start the Ray cluster using a single command through the [cluster launcher](https://docs.ray.io/en/latest/cluster/launcher.html) but we could not get this to work on the Raspberry Pi cluster.
        - In the case of Ray, the environment variables from `./src/env.sh` are read at the time you set up the cluster and not at the time you start the wrapper_script! So make sure to restart the cluster if you adapt values in `./src/env.sh` in between runs. 
    - **For PyTorch DDP / Horovod approaches:** Uncomment relevant section in `./src/train_eval_test.sh`. This is the script that will be executed by the MPI backend on each worker. Make sure, path to environment matches your local settings.
6. Start training by running `./src/wrapper_script.sh` from your head node. Ensure to start this script from the repository root since it is not stable enough to deal with changing relative paths if you start it from e.g. the `src` directory.

## Run single desktop PC approach:

1. Adapt `./src/env.sh` to reflect desired training parameters, specifically set parallelism to 1.
2. Uncomment section `SINGLE DESKTOP PC` in `./src/wrapper_script.sh`. Make sure, path to environment matches your local settings.
3. Start training by running `./src/wrapper_script.sh`. Ensure to start this script from the repository root since it is not stable enough to deal with changing relative paths if you start it from e.g. the `src` directory.

## Running approaches with CPU profiling

To activate the additional CPU profiling, make sure to uncomment the lines starting with sar in `wrapper_script.sh` and `train_eval_test.sh`. Additionally, you need to install and activate [sysstat](https://github.com/sysstat/sysstat) (`sudo apt-get install sysstat`) on all nodes.

## Results

Results are written to the log directory specified in the wrapper script; per default this is `../logs/START_TIMESTAMP` so this is created one level above the repository directory.

This includes:
- the current repository state with commit status, hash, and changes **to be able to completely reproduce the exact setup later**
- logs containing e.g. evaluation results against the validation and test dataset reporting loss and accuracy
- CPU profiling results

Please note that the `../logs` directory will be created on the head node only in the case of the Ray approaches and on every node for the PyTorch DDP and Horovod approaches where it will contain some node-specific logs. However, the most interesting/relevant information can be found on the head node.

Note that the results from the asynchronous parameter server approach are not fully reproducible because the order of model weight updates is determined by e.g. network latency that cannot be controlled by setting a random seed!

## *Optional:* Running PyTorch DDP or Horovod approaches locally

Thanks to Jakob & Torben for this approach to run everything on a `docker-compose` cluster:

1. `docker-compose up -d --build` (This might take a while, since pytorch with MPI support needs to be built)
2. `ssh -i id_rsa pi@localhost` to start an SSH session in the master node. Password is `raspberry`.
3. `ssh-keyscan slave1 > .ssh/known_hosts` and `ssh-keyscan slave2 >> .ssh/known_hosts` to allow future SSH connections from the `master` container to the `slave` container without a (yes/no) prompt
4. `mpirun --host master,slave hostname` to check if the connection is set up correctly (this should print `master` and `slave`)

Then, you can run the PyTorch DDP and Horovod approaches using the `mpirun` command. Ray does not need this setup.
