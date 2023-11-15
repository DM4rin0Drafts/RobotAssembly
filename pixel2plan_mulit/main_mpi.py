from mpi4py import MPI
import numpy as np
import torch

from simulation.masterMPI import MasterMPI
from simulation.workerMPI import WorkerMPI
from simulation.dispatcher import DispatcherMPI
from environment.token import Token
from utilities.argumentparser import *
from simulation import MPIRank
import random
import pydevd_pycharm

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

torch.autograd.set_detect_anomaly(True)

# TODO actor.eval() and actor.train() (https://jamesmccaffrey.wordpress.com/2019/01/23/pytorch-train-vs-eval-mode/)
# TODO CUDA MPI https://devblogs.nvidia.com/introduction-cuda-aware-mpi/

def main():
    torch.manual_seed(rank * 1000)
    np.random.seed(rank * 1000)
    random.seed(rank * 1000)
    args = parse_args()

    # TODO: This has to be defined somewhere else and also only supports 1 token
    args.n_features_actor = 4
    args.dim_single_pred_actor = 3
    # TODO: This formula only supports 1 action at maximum per token and does not consider the order of actions performed
    args.n_features_critic = args.n_features_actor + args.dim_single_pred_actor
    args.dim_single_pred_critic = 1
    args.n_layers = 3
    args.embed_dim = 64
    args.n_heads = 4
    args.n_hidden_readout = []  # TODO: Add hidden layer dimensions here
    args.n_episodes = 5

    if args.debug:
        debug_port_mapping = [1193, 1194, 1192]
        pydevd_pycharm.settrace('localhost', port=debug_port_mapping[rank], stdoutToServer=True, stderrToServer=True)

    if rank == MPIRank.MASTER:
        device_master = "cuda:0" if torch.cuda.is_available() else "cpu"
        master = MasterMPI(args, demo=False, device=device_master)
        master.run()
    elif rank == MPIRank.DISPATCHER:
        dispatcher = DispatcherMPI()
        dispatcher.run()
    else:
        device_worker = "cpu"
        worker = WorkerMPI(args, worker_id=rank, device=device_worker)
        worker.run()


if __name__ == '__main__':
    main()
