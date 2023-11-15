from mpi4py import MPI

from simulation.master import Master
from . import MPITag, MPIRank

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


class MasterMPI(Master):

    def __init__(self, args, demo=False, device="cpu"):
        self.max_pending_sims = args.max_pending_sims
        self.batch_size = args.batch_size

        super(MasterMPI, self).__init__(args, demo, device)

    @property
    def pending_cnt(self):
        return self.given_jobs - self.finished_jobs

    def send_job(self):
        job_id = self.given_jobs
        comm.send(obj=(job_id, self.actor.state), dest=MPIRank.DISPATCHER, tag=MPITag.SEND_POLICY)
        self.given_jobs += 1

    def iteration(self):
        # Multi-processing scheme: Initialize max number of parallel rollouts as allowed
        while self.given_jobs < self.n_episodes and self.pending_cnt < self.max_pending_sims:
            self.send_job()

        # Receive simulation results from dispatcher:
        probe = comm.iprobe(source=MPIRank.DISPATCHER, tag=MPITag.SEND_REWARDS)
        if probe:
            # Receive result:
            result = comm.recv(source=MPIRank.DISPATCHER, tag=MPITag.SEND_REWARDS)
            job_id, trajectories = result
            print(f"[Master    ] Received result for job {job_id}. Adding trajectories to pool")
            self.ddpg.actor.episode_count = self.finished_jobs

            self.replaybuffer.add(trajectories)
            self.ddpg.train(0, self.replaybuffer, self.batch_size, writer=self.writer, t_step=self.finished_jobs)        # TODO remove idx = 0

            self.finished_jobs += 1

            if (self.finished_jobs % 10) == 0:
                self.test(self.finished_jobs)

    def iterate_without_update(self):
        # Multi-processing scheme: Initialize max number of parallel rollouts as allowed
        while self.given_jobs < 100 and self.pending_cnt < self.max_pending_sims:
            self.send_job()

        # Receive simulation results from dispatcher:
        probe = comm.iprobe(source=MPIRank.DISPATCHER, tag=MPITag.SEND_REWARDS)
        if probe:
            # Receive result:
            result = comm.recv(source=MPIRank.DISPATCHER, tag=MPITag.SEND_REWARDS)
            job_id, trajectories = result
            print(f"[Master    ] Received result for job {job_id}. Adding trajectories to pool")
            self.replaybuffer.add(trajectories)

            self.finished_jobs += 1

    def initialize_replay_buffer(self):
        while self.finished_jobs < 100:
            self.iterate_without_update()

        self.given_jobs = 0
        self.finished_jobs = 0

    def run(self):
        self.test(t_step=self.finished_jobs)  # test once before training

        super(MasterMPI, self).run()  # train

        self.test(self.finished_jobs)  # test once after training

        # Terminate the dispatcher (triggers termination cascade to all processes)
        comm.send(obj=None, dest=MPIRank.DISPATCHER, tag=MPITag.TERMINATE)

        super(MasterMPI, self).save()

        print("[Master    ] Terminating")
