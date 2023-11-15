from typing import List

from mpi4py import MPI

from environment import Token
from simulation.master import Master
from . import MPITag, MPIRank
from .training_setup import TrainingSetup

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


class MasterMPI(Master):
    def __init__(self, tokens: List[Token], train_setup: TrainingSetup, run: str or None, gui: False, demo=False,
                 device="cpu"):
        super(MasterMPI, self).__init__(tokens, train_setup, run, gui, demo, device)

        self.max_pending_sims = train_setup.max_pending_sims
        self.batch_size = train_setup.batch_size

    @property
    def n_pending_jobs(self):
        return self.n_given_jobs - self.n_finished_jobs

    def send_job(self):
        job_id = self.n_given_jobs
        comm.send(obj=(job_id, self.actor.state), dest=MPIRank.DISPATCHER, tag=MPITag.SEND_POLICY)
        self.n_given_jobs += 1

    def iteration(self):
        # Multi-processing scheme: Initialize max number of parallel rollouts as allowed
        while self.n_given_jobs < self.n_episodes and self.n_pending_jobs < self.max_pending_sims:
            self.send_job()

        # Receive simulation results from dispatcher:
        probe = comm.iprobe(source=MPIRank.DISPATCHER, tag=MPITag.SEND_REWARDS)
        if probe:
            # Receive result:
            result = comm.recv(source=MPIRank.DISPATCHER, tag=MPITag.SEND_REWARDS)
            job_id, trajectories = result
            print(f"[Master    ] Received result for job {job_id}. Adding trajectories to pool")
            self.ddpg.actor.episode_count = self.n_finished_jobs

            self.replaybuffer.add(trajectories)
            self.ddpg.train(self.replaybuffer, self.batch_size, writer=self.writer, t_step=self.n_finished_jobs)

            self.n_finished_jobs += 1

            if (self.n_finished_jobs % 10) == 0:
                self.test(self.n_finished_jobs)

    def iterate_without_update(self):
        # Multi-processing scheme: Initialize max number of parallel rollouts as allowed
        while self.n_given_jobs < 100 and self.n_pending_jobs < self.max_pending_sims:
            self.send_job()

        # Receive simulation results from dispatcher:
        probe = comm.iprobe(source=MPIRank.DISPATCHER, tag=MPITag.SEND_REWARDS)
        if probe:
            # Receive result:
            result = comm.recv(source=MPIRank.DISPATCHER, tag=MPITag.SEND_REWARDS)
            job_id, trajectories = result
            print(f"[Master    ] Received result for job {job_id}. Adding trajectories to pool")
            self.replaybuffer.add(trajectories)

            self.n_finished_jobs += 1

    def initialize_replay_buffer(self):
        while self.n_finished_jobs < 100:
            self.iterate_without_update()

        self.n_given_jobs = 0
        self.n_finished_jobs = 0

    def run(self):
        self.test(t_step=self.n_finished_jobs)  # test once before training

        super(MasterMPI, self).run()  # train

        self.test(self.n_finished_jobs)  # test once after training

        # Terminate the dispatcher (triggers termination cascade to all processes)
        comm.send(obj=None, dest=MPIRank.DISPATCHER, tag=MPITag.TERMINATE)

        super(MasterMPI, self).save(n_ckpt=0)

        print("[Master    ] Terminating")
