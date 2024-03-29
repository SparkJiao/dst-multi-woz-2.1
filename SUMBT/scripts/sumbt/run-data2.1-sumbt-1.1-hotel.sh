#!/bin/bash

#output_dir=exp-multiwoz/data2.1-sumbt/v1.0
output_dir=exp-multiwoz/data2.1-sumbt/v1.1-hotel  # lr 5e-5 -> 1e-4
target_slot='hotel'
bert_dir='/home/jiaofangkai/'

python3 code/main-multislot-f-5.py --do_train --do_eval --num_train_epochs 100 --data_dir data/multiwoz2.1_5 \
--bert_model bert-base-uncased --do_lower_case --bert_dir $bert_dir --task_name bert-gru-sumbt --nbt rnn \
--output_dir $output_dir --target_slot ${target_slot} --warmup_proportion 0.1 --learning_rate 1e-4 \
--train_batch_size 4 --eval_batch_size 4 --distance_metric euclidean --patience 15 \
--train_file data/multiwoz2.1_5/train-5-full-value.tsv \
--dev_file data/multiwoz2.1_5/dev-5-full-value.tsv \
--test_file data/multiwoz2.1_5/test-5-full-value.tsv \
--ontology data/multiwoz2.1_5/ontology-full.json \
--tf_dir tensorboard --hidden_dim 300 --max_label_length 20 --max_seq_length 64 --max_turn_length 22 --max_slot_length 6

target_slot='all'

python3 code/main-multislot-f-5.py --do_eval --num_train_epochs 100 --data_dir data/multiwoz2.1_5 \
--bert_model bert-base-uncased --do_lower_case --bert_dir $bert_dir --task_name bert-gru-sumbt --nbt rnn \
--output_dir $output_dir --target_slot ${target_slot} --warmup_proportion 0.1 --learning_rate 1e-4 \
--train_batch_size 4 --eval_batch_size 4 --distance_metric euclidean --patience 15 \
--train_file data/multiwoz2.1_5/train-5-full-value.tsv \
--dev_file data/multiwoz2.1_5/dev-5-full-value.tsv \
--test_file data/multiwoz2.1_5/test-5-full-value.tsv \
--ontology data/multiwoz2.1_5/ontology-full.json \
--tf_dir tensorboard --hidden_dim 300 --max_label_length 20 --max_seq_length 64 --max_turn_length 22 --max_slot_length 6

