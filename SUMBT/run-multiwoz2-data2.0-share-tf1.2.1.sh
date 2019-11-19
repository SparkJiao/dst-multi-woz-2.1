#!/bin/bash

#output_dir=exp-multiwoz/exp-data2.0-share-tf1.2.1
output_dir=exp-multiwoz/exp-data2.0-share-tf1.2.2  # Fix dev accuracy calculation error
target_slot='all'
bert_dir='/home/jiaofangkai/'

#conda activate pretrained-bert0.6.2

python code/main-multislot-share.py --do_train --do_eval --num_train_epochs 30 --data_dir data/multiwoz2.0 \
--bert_model bert-base-uncased --do_lower_case --bert_dir $bert_dir --task_name bert-gru-sumbt \
--nbt transformer --output_dir $output_dir --target_slot all --warmup_proportion 0.1 --learning_rate 5e-5 \
--train_batch_size 8 --eval_batch_size 1 --distance_metric product --patience 10 \
--tf_dir tensorboard --max_seq_length 96 --max_turn_length 22 \
--fp16 --fp16_opt_level O2 --gradient_accumulation_steps 8 \
--reduce_layers 2 --num_rnn_layers 4 --max_label_length 17 --max_slot_length 6

# 52.90% joint accuracy
