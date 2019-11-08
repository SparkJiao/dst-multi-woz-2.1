#!/bin/bash

output_dir=exp-multiwoz/exp-fix2-data2.0-F-restaurant
target_slot='restaurant'
bert_dir='/home/jiaofangkai/'

#python3 code/main-multislot-f.py --do_train --do_eval --num_train_epochs 100 --data_dir data/multiwoz2.0 \
#--bert_model bert-base-uncased --do_lower_case --bert_dir $bert_dir --task_name bert-gru-sumbt \
#--nbt rnn --output_dir $output_dir --target_slot $target_slot --warmup_proportion 0.1 --learning_rate 1e-4 \
#--train_batch_size 3 --eval_batch_size 4 --distance_metric euclidean --patience 15 --tf_dir tensorboard \
#--hidden_dim 300 --max_label_length 32 --max_seq_length 64 --max_turn_length 22

target_slot='all'
python3 code/main-multislot-f.py --do_eval --num_train_epochs 100 --data_dir data/multiwoz2.0 \
--bert_model bert-base-uncased --do_lower_case --bert_dir $bert_dir --task_name bert-gru-sumbt \
--nbt rnn --output_dir $output_dir --target_slot $target_slot --warmup_proportion 0.1 --learning_rate 1e-4 \
--train_batch_size 3 --eval_batch_size 4 --distance_metric euclidean --patience 15 --tf_dir tensorboard \
--hidden_dim 300 --max_label_length 32 --max_seq_length 64 --max_turn_length 22