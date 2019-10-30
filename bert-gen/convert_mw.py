import json
import os

""" Some code comes from https://github.com/renll/ComerNet/blob/master/convert_mw.py """

### this file is to convert the raw woz data into the format required for prepross.py


cldict = {'pricerange': 'price range', 'leaveat': 'leave at', 'arriveby': 'arrive by'}


switch = 0
if __name__ == '__main__':

    # Make vocabulary for domain slot pair
    ontology = 'data/multi-woz/MULTIWOZ2 2/ontology.json'
    with open(ontology, 'r') as f:
        ontology_dict = json.load(f)
    domain_slot_pairs = []
    for key in ontology_dict.keys():
        domain, slot = key.split('-')
        if slot in cldict:
            slot = cldict[slot]
        domain_slot_pairs.append(f'{domain}#{slot}')
    domain_slot_vocab = {ds: idx for idx, ds in enumerate(domain_slot_pairs)}
    print(f'total domain slot pairs: {len(domain_slot_vocab)}')
    with open('data/domain_slot_vocab.json', 'w') as f:
        json.dump(domain_slot_vocab, f, indent=2)

    for mode in ['train', 'dev', 'test']:
        fp = os.path.join('data/', '{}_dials.json'.format(mode))
        with open(fp, 'r') as json_file:
            data = json.load(json_file)

        data_input = []
        for s in data:
            s_dict = {
                'user_input': [],
                'system_input': [],
                'belief_state': []
            }

            for i, turn in enumerate(s['dialogue']):
                svs = {}
                for st in turn['belief_state']:  # loop slot-value pairs
                    if st['act'] == 'inform':
                        dsl, val = st['slots'][0][0], st['slots'][0][1]
                        if val == 'dontcare':
                            val = "do not care"
                        dm = dsl.split('-')[0]
                        sl = dsl.split('-')[1]

                        if sl in list(cldict.keys()):
                            sl = cldict[sl]
                        ds = f'{dm}#{sl}'
                        if ds not in domain_slot_vocab:
                            print(f'{ds} not in domain-slot pairs vocabulary')
                            continue
                        svs[ds] = val

                s_dict['belief_state'].append(svs)

                a = turn['system_transcript']
                b = turn['transcript']

                s_dict['system_input'].append(a)
                s_dict['user_input'].append(b)

            data_input.append(s_dict)

        print(len(data_input))

        with open(f'data/format_{mode}.json', 'w') as f:
            json.dump(data_input, f, indent=2)
