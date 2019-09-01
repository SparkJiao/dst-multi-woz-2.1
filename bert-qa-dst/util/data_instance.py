from enum import Enum, unique
from typing import Dict, List


@unique
class State(Enum):
    TRAIN = 777
    VALIDATE = 7777
    TEST = 77777


class DialogTurnExample:
    def __init__(self, usr_utt: str, sys_utt: str, domain_slot_value: Dict[str, str]):
        self.usr_utt = usr_utt
        self.sys_utt = sys_utt
        self.domain_slot_value = domain_slot_value


class MultiWOZExample:
    def __init__(self, dialog_id: str, dialog_turn_examples: List[DialogTurnExample]):
        self.dialog_id = dialog_id
        self.dialog_turn_examples = dialog_turn_examples
