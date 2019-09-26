import json
from collections import OrderedDict
import logging

from util.config import args, EXPERIMENT_DOMAINS
from util.fix_label import fix_general_label_error
from util import multi_woz_utils

logger = logging.getLogger(__name__)


class MultiWOZReader:
    def __init__(self, ontology_file):
        self.gating_vocab = {
            "ptr": 0,
            "dontcare": 1,
            "none": 2
        }
        self.SLOTS = multi_woz_utils.get_slot_information(json.load(open(ontology_file, 'r')))

    def read_train(self, input_file):
        pass

    def read_dev(self, input_file):
        pass

    def read_test(self, input_file):
        pass

    def _read(self, input_file, state: str):
        with open(input_file, 'r') as f:
            dials = json.load(f)

            for dial_dict in dials:
                # Unseen domain setting
                if args.only_domain != '' and args.only_domain not in dial_dict["domains"]:
                    continue
                if (args.except_domain != '' and state == 'test' and args.except_domain not in dial_dict["domains"]) or \
                        (args.except_domain != '' and state != 'test' and args.except_domain in dial_dict["domains"]):
                    continue

                # Reading data
                for ti, turn in enumerate(dial_dict["dialogue"]):
                    turn_domain = turn["domain"]
                    turn_id = turn["turn_idx"]
                    turn_utt = turn["system_transcript"] + " ; " + turn["transcript"]
                    turn_utt_strip = turn_utt.strip()
                    turn_belief_dict = fix_general_label_error(turn["belief_state"], False, self.SLOTS)

                    # Generate domain-dependent slot list
                    slot_temp = self.SLOTS
                    if state == "train" or state == "dev":
                        if args.except_domain != "":
                            slot_temp = [k for k in self.SLOTS if args.except_domain not in k]
                            turn_belief_dict = OrderedDict(
                                [(k, v) for k, v in turn_belief_dict.items() if args.except_domain not in k])
                        elif args.only_domain != "":
                            slot_temp = [k for k in self.SLOTS if args.only_domain in k]
                            turn_belief_dict = OrderedDict([(k, v) for k, v in turn_belief_dict.items() if args.only_domain in k])
                    else:
                        if args.except_domain != "":
                            slot_temp = [k for k in self.SLOTS if args.except_domain in k]
                            turn_belief_dict = OrderedDict([(k, v) for k, v in turn_belief_dict.items() if args.except_domain in k])
                        elif args.only_domain != "":
                            slot_temp = [k for k in self.SLOTS if args.only_domain in k]
                            turn_belief_dict = OrderedDict([(k, v) for k, v in turn_belief_dict.items() if args.only_domain in k])

                    turn_belief_list =
