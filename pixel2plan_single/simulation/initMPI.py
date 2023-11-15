from enum import IntEnum, auto


class MPITag(IntEnum):
    TERMINATE = auto()
    SEND_POLICY = auto()
    SEND_REWARDS = auto()


class MPIRank(IntEnum):
    MASTER = 0
    DISPATCHER = 1
    WORKER = 2  # and greater