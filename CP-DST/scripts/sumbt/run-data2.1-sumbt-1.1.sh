#!/bin/bash

output_dir=exp-multiwoz/data2.1-sumbt/v1.1
target_slot='all'
bert_dir='/home/cp-dst/'

python3 code/main-multislot-f-5.py --do_eval --num_train_epochs 100 --data_dir data/multiwoz2.1_5 \
--bert_model bert-base-uncased --do_lower_case --bert_dir $bert_dir --task_name bert-gru-sumbt --nbt rnn \
--output_dir $output_dir --target_slot all --warmup_proportion 0.1 --learning_rate 1e-4 \
--train_batch_size 4 --eval_batch_size 4 --distance_metric euclidean --patience 15 \
--train_file data/multiwoz2.1_5/train-5-full-value.tsv \
--dev_file data/multiwoz2.1_5/dev-5-full-value.tsv \
--test_file data/multiwoz2.1_5/test-5-full-value.tsv \
--ontology data/multiwoz2.1_5/ontology-full.json \
--tf_dir tensorboard --hidden_dim 300 --max_label_length 20 --max_seq_length 64 --max_turn_length 22 --max_slot_length 6

