#!/bin/bash

#output_dir=exp-multiwoz/data2.1-cls-context-stack2/v1.0
#output_dir=exp-multiwoz/data2.1-cls-context-stack2/v1.1  # epoch 5 -> 4 lr 3.5e-5 -> 4e-5
output_dir=exp-multiwoz/data2.1-cls-context-stack2/v1.5  # --max_seq_len 64 -> 108
target_slot='all'
nbt='context_stacked2'
bert_dir='/home/jiaofangkai/'

python code/main-multislot-share-5-newcls.py --do_train --do_eval --num_train_epochs 4 --data_dir data/multiwoz2.1_5 \
--bert_model bert-base-uncased --do_lower_case --bert_dir $bert_dir --task_name bert-gru-sumbt \
--nbt $nbt --output_dir $output_dir --target_slot $target_slot --warmup_proportion 0.1 --learning_rate 4e-5 \
--train_batch_size 1 --eval_batch_size 1 --distance_metric product --patience 5 \
--tf_dir tensorboard --max_seq_length 108 --max_turn_length 22 \
--fp16 --fp16_opt_level O2 --gradient_accumulation_steps 1 \
--reduce_layers 0 --max_label_length 20 --max_slot_length 6 \
--share_position_weight --self_attention_type 1 \
--train_file data/multiwoz2.1_5/train-5-full-value.tsv \
--dev_file data/multiwoz2.1_5/dev-5-full-value.tsv \
--test_file data/multiwoz2.1_5/test-5-full-value.tsv \
--ontology data/multiwoz2.1_5/ontology-full.json \
--extra_nbt --fuse_type 1 --cls_type 0 \
--context_add_residual --context_add_layer_norm --ff_add_residual --ff_add_layer_norm --query_layer_norm --query_residual \
--override_attn