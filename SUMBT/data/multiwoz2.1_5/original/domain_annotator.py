from tqdm import tqdm
import json

EXPERIMENT_DOMAINS = ['train', 'hotel', 'attraction', 'taxi', 'restaurant']

ignore_keys_in_goal = ['eod', 'messageLen', 'message', 'topic', 'police', 'hospital']
data_file = 'data.json'

domain_list = {dom: [] for dom in EXPERIMENT_DOMAINS}

with open(data_file, 'r') as f:
    data = json.load(f)

for k, v in tqdm(data.items()):
    assert 'goal' in v
    assert 'log' in v
    dialog_goal = v['goal']
    domains = []
    for dom_k, dom_v in dialog_goal.items():
        if dom_v and dom_k not in ignore_keys_in_goal:
            domains.append(dom_k)
    domains = list(set(domains))
    if not ('MUL' in k or 'mul' in k):
        assert len(domains) in [0, 1], (k, domains)
    for dom in domains:
        domain_list[dom].append(k)

with open('../domain_list.json', 'w') as f:
    json.dump(domain_list, f, indent=2)

for dom, dom_ls in domain_list.items():
    print(dom, len(dom_ls))
    print(dom, len(list(set(dom_ls))))
