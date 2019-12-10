#!/bin/bash

#output_dir=exp-multiwoz/data2.0-share-sa-extend1.0  #
#output_dir=exp-multiwoz/data2.0-share-sa-extend1.0-fixbug  # fix the bug of key type 0 and replace masked values with -50000
#output_dir=exp-multiwoz/data2.0-share-sa-extend-key0-1.1  # num_train_epochs 6 -> 8
output_dir=exp-multiwoz/data2.0-share-sa-extend1.0.1  # masked_values -50000 -> -10000 num_train_epochs 8 --learning_rate 5e-5 -> 4e-5

target_slot='all'
nbt='extend'
bert_dir='/home/jiaofangkai/'

python code/main-multislot-share.py --do_train --do_eval --num_train_epochs 8 --data_dir data/multiwoz2.0 \
--bert_model bert-base-uncased --do_lower_case --bert_dir $bert_dir --task_name bert-gru-sumbt \
--nbt $nbt --output_dir $output_dir --target_slot $target_slot --warmup_proportion 0.1 --learning_rate 4e-5 \
--train_batch_size 1 --eval_batch_size 1 --distance_metric product --patience 5 \
--tf_dir tensorboard --max_seq_length 96 --max_turn_length 22 \
--fp16 --fp16_opt_level O2 --gradient_accumulation_steps 1 \
--reduce_layers 0 --max_label_length 17 --max_slot_length 6 \
--override_attn --share_position_weight --key_type 0
