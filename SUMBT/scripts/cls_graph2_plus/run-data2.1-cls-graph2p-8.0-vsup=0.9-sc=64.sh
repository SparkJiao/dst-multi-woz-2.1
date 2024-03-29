#!/bin/bash

#output_dir=exp-multiwoz/data2.1-cls-graph2p/v1.0
#output_dir=exp-multiwoz/data2.1-cls-graph2p/v1.1
#output_dir=exp-multiwoz/data2.1-cls-graph2p/v1.2
#output_dir=exp-multiwoz/data2.1-cls-graph2p/v1.3  # lr 4e-5 -> 3.5e-5
#output_dir=exp-multiwoz/data2.1-cls-graph2p/v1.4  # lr 3.5e-5 -> 3e-5
#output_dir=exp-multiwoz/data2.1-cls-graph2p/v1.4-7.0  # --fp16_opt_level O2 -> O1
#output_dir=exp-multiwoz/data2.1-cls-graph2p/v1.4-7.1  # lr 3e-5 -> 4e-5
#output_dir=exp-multiwoz/data2.1-cls-graph2p/v1.4-7.8  # --sa_add_layer_norm
#output_dir=exp-multiwoz/data2.1-cls-graph2p/v1.4-7.14   # --graph_value_sup 0.9 -> 1.0
#output_dir=exp-multiwoz/data2.1-cls-graph2p/v7.0  # adopt multi-head attention in graph
#output_dir=exp-multiwoz/data2.1-cls-graph2p/v7.1  # lr 4e-5 -> 5e-5
#output_dir=exp-multiwoz/data2.1-cls-graph2p/v7.5  # warmup_proportion 0.1 -> 0.2
#output_dir=exp-multiwoz/data2.1-cls-graph2p/v7.5-head=12  # graph_attn_head 6 -> 12
#output_dir=exp-multiwoz/data2.1-cls-graph2p/v8.0  # --graph_add_layer_norm
output_dir=exp-multiwoz/data2.1-cls-graph2p/v8.0-vsup=0.9-scale=64
target_slot='all'
nbt='graph2_p'
bert_dir='/home/jiaofangkai/'

# dev: 56.5365  test: 53.947

python code/main-multislot-share-5-newcls.py --do_train --do_eval --num_train_epochs 5 --data_dir data/multiwoz2.1_5 \
--bert_model bert-base-uncased --do_lower_case --bert_dir $bert_dir --task_name bert-gru-sumbt \
--nbt $nbt --output_dir $output_dir --target_slot $target_slot --warmup_proportion 0.2 --learning_rate 5e-5 \
--train_batch_size 1 --eval_batch_size 1 --distance_metric product --patience 10 \
--tf_dir tensorboard --max_seq_length 108 --max_turn_length 22 \
--fp16 --fp16_opt_level O1 --gradient_accumulation_steps 1 \
--reduce_layers 0 --max_label_length 20 --max_slot_length 6 \
--share_position_weight --self_attention_type 1 \
--train_file data/multiwoz2.1_5/train-5-full-value.tsv \
--dev_file data/multiwoz2.1_5/dev-5-full-value.tsv \
--test_file data/multiwoz2.1_5/test-5-full-value.tsv \
--ontology data/multiwoz2.1_5/ontology-full.json \
--cls_type 0 --extra_nbt --graph_value_sup 0.9 --attn_head 12 --extra_nbt_attn_head 12 --diag_attn_hidden_scale 1.0 \
--max_loss_scale 64 --value_embedding_type mean --sa_add_layer_norm --graph_attn_type 1 --graph_attn_head 12 --graph_add_layer_norm