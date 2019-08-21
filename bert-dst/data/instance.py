from enum import Enum, unique
from typing import Dict, List

@unique
class State(Enum):
    Train = 0
    Evaluation = 1
    Test = 2


class MultiWOZInputExample:
    def __init__(self,
                 dialog_id: str,
                 turn_index: int,
                 user_utt_tokens: str,
                 system_utt_tokens: str,
                 user_utt_char2tok_map: List[int],
                 system_utt_char2tok_map: List[int],
                 domain_slot_value: Dict[str, Dict]):
        self.dialog_id = dialog_id
        self.turn_index = turn_index
        self.user_utt_tokens = user_utt_tokens
        self.system_utt_tokens = system_utt_tokens
        self.user_utt_char2tok_map = user_utt_char2tok_map
        self.system_utt_char2tok_map = system_utt_char2tok_map
        self.domain_slot_value = domain_slot_value


class InputFeature:
    def __init__(self):
        pass
