# pytorch-distributed-rnn

## Running the `DistributedDataParralel` example on a `docker-compose` cluster

1. `docker-compose up -d --build` (This might take a while, since pytorch with MPI support needs to be built)
2. `ssh -i id_rsa pi@localhost` to start an SSH session in the master node. Password is `raspberry`.
3. `ssh-keyscan slave > .ssh/known_hosts` to allow future SSH connections from the `master` container to the `slave` container without a (yes/no) prompt
4. `mpirun --host master,slave hostname` to check if the connection is set up correctly (this should print `master` and `slave`)
5. `mpirun --host master,slave python3 src/example/example_ddp.py` (the final parameters of the model should be the same for all ranks, which indicates that the distributed training was successful)
