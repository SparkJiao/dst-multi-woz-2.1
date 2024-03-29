#!/bin/bash

output_dir=exp-multiwoz/data2.1-cls-graph6/v1.0-psup=1.0-vsup=0.8  # version 8.0 use transformer as belief tracker --sa_add_residual
target_slot='all'
nbt='graph6'
bert_dir='/home/jiaofangkai/'

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
--cls_type 0 --fuse_type 4 --extra_nbt --extra_nbt_attn_head 12 --graph_value_sup 0.8 --attn_head 12 --diag_attn_hidden_scale 1.0 \
--max_loss_scale 64 --value_embedding_type mean --sa_add_layer_norm --sa_add_residual --mask_self \
--graph_attn_type 0 --graph_attn_head 12 --graph_add_layer_norm --graph_add_output --sa_fuse_type gate --pre_cls_sup 1.0