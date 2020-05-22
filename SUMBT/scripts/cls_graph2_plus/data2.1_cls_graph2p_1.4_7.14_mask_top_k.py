import logging
import os
import subprocess
import argparse
import time

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def run_cmd(cmd: str):
    logger.info(cmd)
    subprocess.check_call(cmd, shell=True)


output_dir = 'exp-multiwoz/data2.1-cls-graph2p/v1.4-7.14'  # --graph_value_sup 0.9 -> 1.0
target_slot = 'all'
nbt = 'graph2_p_test_mask'
bert_dir = '/home/jiaofangkai/'

cmd = f'python code/main-multislot-share-5-newcls.py --do_eval --num_train_epochs 5 --data_dir data/multiwoz2.1_5 \
--bert_model bert-base-uncased --do_lower_case --bert_dir {bert_dir} --task_name bert-gru-sumbt \
--nbt {nbt} --output_dir {output_dir} --target_slot {target_slot} --warmup_proportion 0.1 --learning_rate 4e-5 \
--train_batch_size 1 --eval_batch_size 1 --distance_metric product --patience 5 \
--tf_dir tensorboard --max_seq_length 108 --max_turn_length 22 \
--fp16 --fp16_opt_level O1 --gradient_accumulation_steps 1 \
--reduce_layers 0 --max_label_length 20 --max_slot_length 6 \
--share_position_weight --self_attention_type 1 \
--train_file data/multiwoz2.1_5/train-5-full-value.tsv \
--dev_file data/multiwoz2.1_5/dev-5-full-value.tsv \
--test_file data/multiwoz2.1_5/test-5-full-value.tsv \
--ontology data/multiwoz2.1_5/ontology-full.json \
--cls_type 0 --extra_nbt --graph_value_sup 1.0 --attn_head 12 --extra_nbt_attn_head 12 --diag_attn_hidden_scale 1.0 \
--max_loss_scale 256 --value_embedding_type mean --sa_add_layer_norm --save_gate'

for mode in range(35, 38):
    for i in range(1, 2):
        mask_top_k = i
        # predict_dir = f'{output_dir}/mask_top_{mask_top_k}_test_mode_{mode}'
        predict_dir = f'{output_dir}/test_mode_{mode}_reverse'
        cur_cmd = cmd + f' --mask_top_k {mask_top_k} --predict_dir {predict_dir} --test_mode {mode} '
        run_cmd(cur_cmd)
