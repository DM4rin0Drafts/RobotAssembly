"""
The dispatcher is designed with the aim of being highly reactive (i.e., non-blocking).

This is necessary to wait neither for the master (i.e., the reinforcement learning algorithm)
nor for the workers (i.e., the simulations).
"""

from queue import Queue, SimpleQueue
from queue import Empty as QueueEmptyException
from mpi4py import MPI

from . import MPITag, MPIRank

import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


class DispatcherMPI(object):

    def __init__(self):

        self._job_pool = SimpleQueue()
        self._result_pool = SimpleQueue()
        self._request = None

        self.__idling_workers = Queue(maxsize=size)

        super(DispatcherMPI, self).__init__()

    @classmethod
    def terminate_workers(cls):
        # Iterate over all workers to terminate each worker:
        for worker_rank in range(MPIRank.WORKER, size):
            print(f"[Dispatcher] Sending termination signal to Worker {worker_rank}")
            comm.send(obj=None, dest=worker_rank, tag=MPITag.SEND_POLICY)

    def send_results(self):
        # Send results asynchronously to the master if available (thus, non-blocking):
        if self._request is None or MPI.Request.Test(self._request):
            try:
                result = self._result_pool.get_nowait()
                self._request = comm.isend(obj=result, dest=MPIRank.MASTER, tag=MPITag.SEND_REWARDS)

            except QueueEmptyException:
                pass  # Currently no new result available.

    def receive_results(self, recv_status):
        # Receive results (or an initial idle signal if result is None):
        worker_rank = recv_status.Get_source()
        result = comm.recv(source=worker_rank, tag=MPITag.SEND_REWARDS)

        # Add the received result to the result pool:
        if result is not None:
            print(f"[Dispatcher] Received result for job {result[0]}")
            self._result_pool.put_nowait(result)

        # Worker is now idling:
        # print(f"Worker {worker_rank}: idling")
        self.__idling_workers.put_nowait(worker_rank)

    def send_job(self):
        job = self._job_pool.get_nowait()
        worker_rank = self.__idling_workers.get_nowait()
        comm.send(obj=job, dest=worker_rank, tag=MPITag.SEND_POLICY)

    def recv_job(self):
        # Receive job:
        job = comm.recv(source=MPIRank.MASTER, tag=MPITag.SEND_POLICY)

        # Add job to the job pool:
        print(f"[Dispatcher] Received job {job[0]} from Master")
        self._job_pool.put_nowait(job)
        # print(f"dispatcher job pool {self._job_pool.qsize()}")

    def run(self):
        # Run until termination signal is received
        while not comm.iprobe(source=MPIRank.MASTER, tag=MPITag.TERMINATE):

            # Check if a worker has send a reward (non-blocking):
            recv_status = MPI.Status()
            probe = comm.iprobe(tag=MPITag.SEND_REWARDS, status=recv_status)

            # Retrieve the data from the worker:
            if probe:
                self.receive_results(recv_status)

            # Send new job to idling workers if job & idle worker are available (non-blocking):
            if not (self._job_pool.empty() or self.__idling_workers.empty()):
                self.send_job()

            # Check whether the master has send new jobs (non-blocking):
            probe = comm.iprobe(source=MPIRank.MASTER, tag=MPITag.SEND_POLICY)

            if probe:
                self.recv_job()

            self.send_results()

        MPI.Request.Wait(self._request)
        self.terminate_workers()
        print("[Dispatcher] Terminating")
