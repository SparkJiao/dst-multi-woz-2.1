#!/bin/bash

output_dir=exp-multiwoz/NRS-1.0
bert_dir='/home/jiaofangkai/'

python3 code/main-nrs.py --do_train --do_eval --num_train_epochs 50 --data_dir data/multiwoz2.0 --bert_model bert-base-uncased --do_lower_case --bert_dir $bert_dir --task_name bert-gru-sumbt --nbt rnn --output_dir $output_dir --warmup_proportion 0.1 --learning_rate 5e-5 --train_batch_size 3 --eval_batch_size 4 --distance_metric euclidean --patience 15 --tf_dir tensorboard --hidden_dim 300 --max_query_length 64 --max_seq_length 128 --max_turn_length 22

