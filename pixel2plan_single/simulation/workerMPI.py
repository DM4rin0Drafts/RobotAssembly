from typing import List

from mpi4py import MPI

from environment import Token
from simulation.worker import Worker
from . import MPITag, MPIRank
from .training_setup import TrainingSetup

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


class WorkerMPI(Worker):
    def __init__(self, worker_id, tokens: List[Token], train_setup: TrainingSetup, gui=False, device="cpu"):
        super(WorkerMPI, self).__init__(worker_id, tokens, train_setup,  gui, device)

    def run(self):
        # No simulation result available the first time
        sim_result = None

        while True:
            # Send results of last simulation and receive a new job:
            job_data = comm.sendrecv(sendobj=sim_result,
                                     dest=MPIRank.DISPATCHER,
                                     sendtag=MPITag.SEND_REWARDS,
                                     source=MPIRank.DISPATCHER,
                                     recvtag=MPITag.SEND_POLICY)

            # Check whether worker should be terminated:
            if job_data is None:
                break

            # Simulate received roll-out:
            rollout_id, parameters = job_data
            trajectories = self.simulate_rollout(parameters=parameters)

            # Package the answer:
            sim_result = [rollout_id, trajectories]

        print(f"[Worker {rank}  ] Terminating")
