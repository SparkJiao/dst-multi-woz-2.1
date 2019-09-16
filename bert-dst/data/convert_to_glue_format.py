import json
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='multi-woz', help='Data dir of MultiWOZ.')
parser.add_argument('--output_dir', type=str, default='data/multiwoz', help='Output directory of processed multi-woz data,')
args = parser.parse_args()

# Read ontology file
fp_ont = open(os.path.join(args.data_dir, "ontology.json"), "r")
data_ont = json.load(fp_ont)
ontology = {}
for domain_slot in data_ont:
    domain, slot = domain_slot.split('-')
    if domain not in ontology:
        ontology[domain] = {}
    ontology[domain][slot] = {}
    for value in data_ont[domain_slot]:
        ontology[domain][slot][value] = 1
fp_ont.close()

# Read file list (dev and test sets are defined)
dev_file_list = {}
fp_dev_list = open(os.path.join(args.data_dir, "valListFile.json"))
for line in fp_dev_list:
    dev_file_list[line.strip()] = 1

test_file_list = {}
fp_test_list = open(os.path.join(args.data_dir, "testListFile.json"))
for line in fp_test_list:
    test_file_list[line.strip()] = 1

# Read woz logs and write to tsv files

fp_train = open(os.path.join(args.output_dir, "train.tsv"), "w")
fp_dev = open(os.path.join(args.output_dir, "dev.tsv"), "w")
fp_test = open(os.path.join(args.output_dir, "test.tsv"), "w")

fp_train.write('# Dialogue ID\tTurn Index\tUser Utterance\tSystem Response\t')
fp_dev.write('# Dialogue ID\tTurn Index\tUser Utterance\tSystem Response\t')
fp_test.write('# Dialogue ID\tTurn Index\tUser Utterance\tSystem Response\t')

for domain in sorted(ontology.keys()):
    for slot in sorted(ontology[domain].keys()):
        fp_train.write(str(domain) + '-' + str(slot) + '\t')
        fp_dev.write(str(domain) + '-' + str(slot) + '\t')
        fp_test.write(str(domain) + '-' + str(slot) + '\t')

fp_train.write('\n')
fp_dev.write('\n')
fp_test.write('\n')

fp_data = open(os.path.join(args.data_dir, "data.json"), "r")
data = json.load(fp_data)

for file_id in data:
    if file_id in dev_file_list:
        fp_out = fp_dev
    elif file_id in test_file_list:
        fp_out = fp_test
    else:
        fp_out = fp_train

    user_utterance = ''
    system_response = ''
    turn_idx = 0
    for idx, turn in enumerate(data[file_id]['log']):
        if idx % 2 == 0:  # user turn
            user_utterance = data[file_id]['log'][idx]['text']
        else:  # system turn
            user_utterance = user_utterance.replace('\t', ' ')
            user_utterance = user_utterance.replace('\n', ' ')
            user_utterance = user_utterance.replace('  ', ' ')

            system_response = system_response.replace('\t', ' ')
            system_response = system_response.replace('\n', ' ')
            system_response = system_response.replace('  ', ' ')

            fp_out.write(str(file_id))  # 0: dialogue ID
            fp_out.write('\t' + str(turn_idx))  # 1: turn index
            fp_out.write('\t' + str(user_utterance))  # 2: user utterance
            fp_out.write('\t' + str(system_response))  # 3: system response

            belief = {}
            for domain in data[file_id]['log'][idx]['metadata'].keys():
                for slot in data[file_id]['log'][idx]['metadata'][domain]['semi'].keys():
                    value = data[file_id]['log'][idx]['metadata'][domain]['semi'][slot].strip()
                    value = value.lower()
                    if value == '' or value == 'not mentioned' or value == 'not given':
                        value = 'none'

                    if slot == "leaveAt" and domain != "bus":
                        slot = "leave at"
                    elif slot == "arriveBy" and domain != "bus":
                        slot = "arrive by"
                    elif slot == "pricerange":
                        slot = "price range"

                    if value == "doesn't care" or value == "don't care" or value == "dont care" or value == "does not care":
                        value = "do not care"
                    elif value == "guesthouse" or value == "guesthouses":
                        value = "guest house"
                    elif value == "city center" or value == "town centre" or value == "town center" or \
                            value == "centre of town" or value == "center" or value == "center of town":
                        value = "centre"
                    elif value == "west part of town":
                        value = "west"
                    elif value == "mutliple sports":
                        value = "multiple sports"
                    elif value == "swimmingpool":
                        value = "swimming pool"
                    elif value == "concerthall":
                        value = "concert hall"

                    if domain not in ontology:
                        print("domain (%s) is not defined" % domain)
                        continue

                    if slot not in ontology[domain]:
                        print("slot (%s) in domain (%s) is not defined" % (slot, domain))  # bus-arriveBy not defined
                        continue

                    if value not in ontology[domain][slot] and value != 'none':
                        print("%s: value (%s) in domain (%s) slot (%s) is not defined in ontology" %
                              (file_id, value, domain, slot))
                        value = 'none'

                    belief[str(domain) + '-' + str(slot)] = value

                for slot in data[file_id]['log'][idx]['metadata'][domain]['book'].keys():
                    if slot == 'booked':
                        continue
                    if domain == 'bus' and slot == 'people':
                        continue  # not defined in ontology

                    value = data[file_id]['log'][idx]['metadata'][domain]['book'][slot].strip()
                    value = value.lower()

                    if value == '' or value == 'not mentioned' or value == 'not given':
                        value = 'none'
                    elif value == "doesn't care" or value == "don't care" or value == "dont care" or value == "does not care":
                        value = "do not care"

                    if str('book ' + slot) not in ontology[domain]:
                        print("book %s is not defined in domain %s" % (slot, domain))
                        continue

                    if value not in ontology[domain]['book ' + slot] and value != 'none':
                        print("%s: value (%s) in domain (%s) slot (book %s) is not defined in ontology" %
                              (file_id, value, domain, slot))
                        value = 'none'

                    belief[str(domain) + '-book ' + str(slot)] = value

            for domain in sorted(ontology.keys()):
                for slot in sorted(ontology[domain].keys()):
                    key = str(domain) + '-' + str(slot)
                    if key in belief:
                        fp_out.write('\t' + belief[key])
                    else:
                        fp_out.write('\tnone')

            fp_out.write('\n')
            fp_out.flush()

            system_response = data[file_id]['log'][idx]['text']
            turn_idx += 1

fp_train.close()
fp_dev.close()
fp_test.close()