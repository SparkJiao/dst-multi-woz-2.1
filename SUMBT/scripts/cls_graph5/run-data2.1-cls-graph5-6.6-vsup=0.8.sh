#!/bin/bash

#output_dir=exp-multiwoz/data2.1-cls-graph5/v1.0
#output_dir=exp-multiwoz/data2.1-cls-graph5/v1.1  # lr 4e-5 -> 3e-5
#output_dir=exp-multiwoz/data2.1-cls-graph5/v2.0  # head_num 6 -> 12
#output_dir=exp-multiwoz/data2.1-cls-graph5/v2.1  # lr 3e-5 -> 4e-5
#output_dir=exp-multiwoz/data2.1-cls-graph5/v5.0  # version 5.0 remove extra belief tracker
#output_dir=exp-multiwoz/data2.1-cls-graph5/v5.2  # lr 4e-5 -> 3e-5
#output_dir=exp-multiwoz/data2.1-cls-graph5/v6.0  # version 6 use multi-head attention
#output_dir=exp-multiwoz/data2.1-cls-graph5/v6.1  # lr 3e-5 -> 4e-5
#output_dir=exp-multiwoz/data2.1-cls-graph5/v6.3  # lr 4e-5 -> 5e-5 graph attn head 6 -> 12 warmup 0.1 -> 0.2 --extra_nbt
#output_dir=exp-multiwoz/data2.1-cls-graph5/v6.4  # --graph_add_layer_norm
output_dir=exp-multiwoz/data2.1-cls-graph5/v6.6-vsup=0.8  # lr 5e-5 -> 4e-5
target_slot='all'
nbt='graph5'
bert_dir='/home/jiaofangkai/'

# dev: 56.9976 test: 53.120

python code/main-multislot-share-5-newcls.py --do_train --do_eval --num_train_epochs 5 --data_dir data/multiwoz2.1_5 \
--bert_model bert-base-uncased --do_lower_case --bert_dir $bert_dir --task_name bert-gru-sumbt \
--nbt $nbt --output_dir $output_dir --target_slot $target_slot --warmup_proportion 0.2 --learning_rate 4e-5 \
--train_batch_size 1 --eval_batch_size 1 --distance_metric product --patience 10 \
--tf_dir tensorboard --max_seq_length 108 --max_turn_length 22 \
--fp16 --fp16_opt_level O1 --gradient_accumulation_steps 1 \
--reduce_layers 0 --max_label_length 20 --max_slot_length 6 \
--share_position_weight --self_attention_type 1 \
--train_file data/multiwoz2.1_5/train-5-full-value.tsv \
--dev_file data/multiwoz2.1_5/dev-5-full-value.tsv \
--test_file data/multiwoz2.1_5/test-5-full-value.tsv \
--ontology data/multiwoz2.1_5/ontology-full.json \
--cls_type 0 --fuse_type 3 --extra_nbt --extra_nbt_attn_head 12 --graph_value_sup 0.8 --attn_head 12 --diag_attn_hidden_scale 1.0 \
--max_loss_scale 256 --value_embedding_type mean --sa_add_layer_norm --mask_self --graph_attn_type 1 --graph_attn_head 12 --graph_add_layer_norm