import torch
from .fix_label import fix_general_label_error
import json

EXPERIMENT_DOMAINS = ["hotel", "train", "restaurant", "attraction", "taxi"]

gating_dict = {
    "ptr": 0,
    "dontcare": 1,
    "none": 2
}

def read_data(file_name, slots, dataset='train', training=True, max_line = None):
    print(f"Reading from {file_name}")
    data = []
    with open(file_name, 'r') as f:
        dials = json.load(f)

        cnt_lin = 1
        for dial_dict in dials:
            last_belief_dict = {}

            # TODO:
            #  Unseen domain filtering

            # Reading data
            for ti, turn in enumerate(dial_dict):
                turn_domain = turn["domain"]
                turn_id = turn["turn_idx"]
                sys_utt = turn["system_transcript"]
                usr_utt = turn["transcript"]
                turn_belief_dict = fix_general_label_error(turn["belief_state"], False, slots)

                class_label, values, slot_mask, gating_label = [], [], [], []
                for slot in slots:
                    if slot in turn_belief_dict.keys():
                        values.append(turn_belief_dict[slot])

                        if turn_belief_dict[slot] == 'dontcare':
                            gating_label.append(gating_dict["dontcare"])
                        elif turn_belief_dict[slot] == 'none':
                            gating_label.append(gating_dict["none"])
                        else:
                            gating_label.append(gating_dict["ptr"])

                    else:
                        values.append("none")
                        gating_label.append(gating_dict["none"])

                data_detail = {
                    "id": dial_dict["dialogue_idx"],
                    "domains": dial_dict["domains"],
                    "turn_domain": turn_domain,
                    "turn_id": turn_id,
                    "gating_label": gating_label,
                    "sys_utt": sys_utt,
                    "usr_utt": usr_utt,
                    "values": values
                }
                data.append(data_detail)
    return data
