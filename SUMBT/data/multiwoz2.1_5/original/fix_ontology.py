import argparse
import json
from fix_label import fix_general_label_error
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--ontology', type=str, required=True)
parser.add_argument('--output_file', type=str, default='ontology.json')

args = parser.parse_args()

EXPERIMENT_DOMAINS = ["hotel", "train", "restaurant", "attraction", "taxi"]

ontology = json.load(open(args.ontology))

ontology = {k: v for k, v in ontology.items() if k.split('-')[0] in EXPERIMENT_DOMAINS}
print(f"5 domain ontology contains {len(ontology)} domain-slot pairs in total.")

for k, v in ontology.items():
    tmp_values = []
    for value in v:
        if value in ["doesn't care", "don't care", "dont care", "does not care", "do n't care", 'dontcare']:
            tmp_values.append('do not care')
        else:
            tmp_values.append(value)
    if 'do not care' not in tmp_values:
        tmp_values.append('do not care')
    ontology[k] = tmp_values

for v in ontology.values():
    assert 'do not care' in v

slots = list(ontology.keys())

fixed = 0
for slot, values in tqdm(ontology.items(), desc='fixing ontology value error..', total=len(ontology)):
    tmp_values = []
    for v in values:
        label_dict_tmp = {slot: v}
        label_dict = fix_general_label_error(label_dict_tmp, slots)
        assert label_dict_tmp[slot] == v
        if label_dict_tmp[slot] != label_dict[slot]:
            tmp_values.append(label_dict_tmp[slot])
            fixed += 1
        else:
            tmp_values.append(v)
    ontology[slot] = tmp_values

print(f"Fixed {fixed} value errors")

with open(args.output_file, 'w') as f:
    json.dump(ontology, f, indent=2)
