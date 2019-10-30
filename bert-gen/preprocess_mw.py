import argparse
import json
from tqdm import tqdm
import logging

import torch
from transformers.tokenization_bert import BertTokenizer

from util.fix_label import fix_general_label_error_f

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
# Parameters for source setting
parser.add_argument('--train_tgt', default='data/format_train.json',
                    help='Path to the training target data')
parser.add_argument('--valid_tgt', default='data/format_dev.json',
                    help="Path to the validation target data")
parser.add_argument('--test_tgt', default='data/format_test.json',
                    help="Path to the test target data")
parser.add_argument('--domain_slot_vocab', default='data/domain_slot_vocab.json',
                    help="Path to the vocabulary of domain slot pairs")

parser.add_argument('--save_data', default='data/save_data',
                    help="Output file for the prepared data")
# Parameters for pre-process setting
parser.add_argument('--shuffle', type=int, default=0,
                    help="Shuffle data")
parser.add_argument('--seed', type=int, default=3435,
                    help="Random seed")
parser.add_argument('--bert', type=str, default='bert-base-uncased',
                    help='Path to directory of pre-trained bert model including vocabulary file'
                         'or name pre-defined in the name list.')
parser.add_argument('--max_seq_length', type=int, default=128,
                    help='Max length for single turn')

opt = parser.parse_args()
torch.manual_seed(opt.seed)

bert_tokenizer = BertTokenizer.from_pretrained(opt.bert)
domain_slot_pairs = json.load(open(opt.domain_slot_vocab, 'r'))


def save_vocabulary(name, vocab, file):
    print('Saving ' + name + ' vocabulary to \'' + file + '\'...')
    torch.save(vocab, file)


def make_data(src_file):
    src1, src2, src3, tgt, srcv, tgtv = [], [], [], [], [], []
    count, ignored = 0, 0

    logger.info(f'Processing {src_file} ...')
    with open(src_file, 'r') as f:
        src_data = json.load(f)

    all_input_ids = []
    all_token_type_ids = []
    all_attention_mask = []
    all_slot_label = []
    # While running in batch mode, just decoding for domain-slot pairs.
    # all_domain_slot_ids = []
    all_value_ids = []

    input_padding = bert_tokenizer.encode_plus('', add_special_tokens=True, max_length=None)
    assert len(input_padding['input_ids']) == 2
    turn_padding = {
        'input_ids': input_padding['input_ids'] + [0] * (opt.max_seq_length - 2),
        'token_type_ids': input_padding['token_type_ids'] + [0] * (opt.max_seq_length - 2),
        'attention_mask': [1] * 2 + [0] * (opt.max_seq_length - 2),
        'slot_label': [-1] * len(domain_slot_pairs)
    }
    for dialog in tqdm(src_data):

        for sys_utt, usr_utt, belief in zip(dialog['system_input'], dialog['user_input'], dialog['belief_state']):

            belief = fix_general_label_error_f(belief, domain_slot_pairs)
            belief = {ds: v for ds, v in belief.items() if v != 'none'}

            inputs = bert_tokenizer.encode_plus(sys_utt, usr_utt, add_special_tokens=True, max_length=opt.max_seq_length)
            if 'num_truncated_tokens' in inputs and inputs['num_truncated_tokens'] > 0:
                logger.info('Attention! you are cropping tokens (swag task is ok). '
                            'If you are training ARC and RACE and you are poping question + options,'
                            'you need to try to use a bigger max seq length!')
            input_ids, token_type_ids = inputs['input_ids'], inputs['token_type_ids']
            attention_mask = [1] * len(input_ids)

            # Padding
            if len(input_ids) < opt.max_seq_length:
                input_ids = input_ids + [0] * (opt.max_seq_length - len(input_ids))
                token_type_ids = token_type_ids + [0] * (opt.max_seq_length - len(token_type_ids))
                attention_mask = attention_mask + [0] * (opt.max_seq_length - len(attention_mask))

            slot_label = []
            value_ids = []
            for ds in domain_slot_pairs.keys():
                if ds in belief:
                    if belief[ds] == 'do not care':
                        slot_label.append(1)
                        value_ids.append([-1])
                    else:
                        slot_label.append(2)
                        # value_ids.append(bert_tokenizer.encode_plus(belief[ds], add_special_tokens=True))
                        # TODO:
                        #  2019.10.30
                        #  Complete padding process, including dialog padding and values with max length.
                        #  Besides, temporarily stop to think about the difference between our model and TRADE.
                else:
                    slot_label.append(0)
                    value_ids.append([-1])

            # dia_domain_slot_ids = []
            # dia_value_ids = []
            # for ds, v in belief.items():
            #     if v == 'do not care':
            #         continue
            #     dia_domain_slot_ids.append(domain_slot_pairs[ds])
            #     value_inputs = bert_tokenizer.encode_plus(v, add_special_tokens=True, max_length=None)
            #     dia_value_ids.append(value_inputs['input_ids'])

        all_input_ids.append(input_ids)
        all_token_type_ids.append(token_type_ids)
        all_attention_mask.append(attention_mask)
        all_slot_label.append(slot_label)
        all_domain_slot_ids.append(dia_domain_slot_ids)
        all_value_ids.append(dia_value_ids)



    srcF = open(srcFile, 'r')
    for l in srcF:  # for each dialogue
        l = eval(l)
        src1_tmp, src2_tmp, src3_tmp, tgt_tmp, tgt_vtmp, src_vtmp = [], [], [], [], [], []

        # hierarchical input for a whole dialogue with multiple turns
        slines = l['system_input']
        ulines = l['user_input']
        plines = l['belief_input']
        pvlines = l['labeld']
        tlines = l['labels']
        tvlines = l['labelv']

        for sWords, uWords, pWords, tWords, tvWords, pvWords in zip(slines, ulines, plines, tlines, tvlines, pvlines):

            # src vocab
            if bert:
                src1_tmp += [[tgtDicts[w] for w in uWords]]
                src2_tmp += [[tgtDicts[w] for w in sWords]]
                # tgt vocab
            src3_tmp += [[tgtDicts[w] for w in pWords]]
            tt = [tgtDicts[w] for w in pvWords]
            tgt_tmp += [tt]
            tv = [[tgtDicts[w] for w in ws] for ws in tWords]
            tgt_vtmp += [tv]

            tpv = [[[tgtDicts[w] for w in ws] for ws in wss] for wss in tvWords]
            src_vtmp += [tpv]

        count += 1

        src1.append(src1_tmp)
        src2.append(src2_tmp)
        src3.append(src3_tmp)
        srcv.append(src_vtmp)
        tgt.append(tgt_tmp)
        tgtv.append(tgt_vtmp)

    srcF.close()
    print(srcv[:5])

    print('Prepared %d dialogues' %
          (len(src1)))

    return dataset(src1, src2, src3, tgt, tgtv, srcv)
