#!/bin/bash

output_dir=exp-multiwoz/exp-data2.1-share-tf1.0
target_slot='all'
bert_dir='/home/jiaofangkai/'

python3 code/main-multislot-share.py --do_eval --num_train_epochs 20 --data_dir data/multiwoz2.1 \
--bert_model bert-base-uncased --do_lower_case --bert_dir $bert_dir --task_name bert-gru-sumbt \
--nbt transformer --output_dir $output_dir --target_slot all --warmup_proportion 0.1 --learning_rate 5e-5 \
--train_batch_size 4 --eval_batch_size 1 --distance_metric product --patience 10 \
--tf_dir tensorboard --max_seq_length 64 --max_turn_length 22 \
--fp16 --fp16_opt_level O2 --gradient_accumulation_steps 4 \
--attn_head 6 --max_label_length 17 --max_slot_length 6

