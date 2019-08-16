from enum import Enum, unique


@unique
class State(Enum):
    Train = 0
    Evaluation = 1
    Test = 2


class InputExample:
    def __init__(self):
        pass


class InputFeature:
    def __init__(self):
        pass
