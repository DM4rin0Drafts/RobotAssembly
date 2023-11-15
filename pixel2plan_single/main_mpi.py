from mpi4py import MPI
import numpy as np
import torch

from environment import Token
from simulation.masterMPI import MasterMPI
from simulation.training_setup import TrainingSetup, Reward, Architecture
from simulation.workerMPI import WorkerMPI
from simulation.dispatcher import DispatcherMPI
from utilities.argumentparser import *
from simulation import MPIRank
import random
import pydevd_pycharm

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

torch.autograd.set_detect_anomaly(True)


# TODO CUDA MPI https://devblogs.nvidia.com/introduction-cuda-aware-mpi/

def main():
    args = parse_args()

    torch.manual_seed(rank * 1000 + args.seed)
    np.random.seed(rank * 1000 + args.seed)
    random.seed(rank * 1000 + args.seed)

    reward_func = Reward.DENSE if args.dense else Reward.SPARSE

    if args.architecture == "FC_Small":
        architecture = Architecture.FC_Small
    elif args.architecture == "FC_Large":
        architecture = Architecture.FC_Large
    elif args.architecture == "MHA":
        architecture = Architecture.MHA
    else:
        print(f"Unknown Architecture \"{args.architecture}\". Exiting")
        return

    train_setup = TrainingSetup(args.n_episodes, args.max_pending_sims, args.n_iterations, args.batch_size,
                                args.buffer_size, args.lr_actor, args.lr_critic, args.log_std, args.gamma, args.tau,
                                reward_func, args.compare_rewards, architecture, args.overfit)

    if args.debug:
        debug_port_mapping = [22988, 22997, 23004]
        pydevd_pycharm.settrace('localhost', port=debug_port_mapping[rank], stdoutToServer=True, stderrToServer=True)

    tokens = [Token(''.join(args.tokens[i])) for i in range(len(args.tokens))]
    if rank == MPIRank.MASTER:
        device_master = "cuda:0" if torch.cuda.is_available() else "cpu"
        master = MasterMPI(tokens, train_setup, args.load, args.gui, demo=False, device=device_master)
        master.run()
    elif rank == MPIRank.DISPATCHER:
        dispatcher = DispatcherMPI()
        dispatcher.run()
    else:
        device_worker = "cpu"
        worker = WorkerMPI(worker_id=rank, tokens=tokens, train_setup=train_setup, gui=args.gui,
                           device=device_worker)
        worker.run()


if __name__ == '__main__':
    main()
