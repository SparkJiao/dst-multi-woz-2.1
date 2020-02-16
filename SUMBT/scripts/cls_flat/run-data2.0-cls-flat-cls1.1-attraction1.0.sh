#!/bin/bash

output_dir=exp-multiwoz/data2.0-cls-sa-flat/v1.1-attraction
target_slot='attraction'
nbt='flat_test1'
bert_dir='/home/jiaofangkai/'

python code/main-multislot-share-cls.py --do_train --do_eval --num_train_epochs 6 --data_dir data/multiwoz2.0 \
--bert_model bert-base-uncased --do_lower_case --bert_dir $bert_dir --task_name bert-gru-sumbt \
--nbt $nbt --output_dir $output_dir --target_slot $target_slot --warmup_proportion 0.1 --learning_rate 5e-5 \
--train_batch_size 1 --eval_batch_size 1 --distance_metric product --patience 5 \
--tf_dir tensorboard --max_seq_length 64 --max_turn_length 22 \
--fp16 --fp16_opt_level O2 --gradient_accumulation_steps 1 \
--reduce_layers 0 --max_label_length 20 --max_slot_length 6 \
--override_attn --share_position_weight --self_attention_type 1 \
--cls_type 1

target_slot='all'

python code/main-multislot-share-cls.py --do_eval --num_train_epochs 6 --data_dir data/multiwoz2.0 \
--bert_model bert-base-uncased --do_lower_case --bert_dir $bert_dir --task_name bert-gru-sumbt \
--nbt $nbt --output_dir $output_dir --target_slot $target_slot --warmup_proportion 0.1 --learning_rate 5e-5 \
--train_batch_size 1 --eval_batch_size 1 --distance_metric product --patience 5 \
--tf_dir tensorboard --max_seq_length 64 --max_turn_length 22 \
--fp16 --fp16_opt_level O2 --gradient_accumulation_steps 1 \
--reduce_layers 0 --max_label_length 20 --max_slot_length 6 \
--override_attn --share_position_weight --self_attention_type 1 \
--cls_type 1
