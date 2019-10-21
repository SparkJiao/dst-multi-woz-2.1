#!/bin/bash

target_slot='all'
bert_dir='/home/jiaofangkai/'

output_dir=exp-multiwoz/stage2.0-NRS-1.0

python3 code/main-nrs.py --do_train --do_eval --num_train_epochs 5 --data_dir data/multiwoz2.0 --bert_model bert-base-uncased --do_lower_case \
--bert_dir $bert_dir --task_name bert-gru-sumbt --nbt rnn --output_dir $output_dir --warmup_proportion 0.01 --learning_rate 1e-4 \
--train_batch_size 32 --dev_batch_size 32 --distance_metric product --patience 15 --tf_dir tensorboard --hidden_dim 300 \
--max_query_length 64 --max_seq_length 64 --max_turn_length 22 --gradient_accumulation_steps 1 \
--model_id 1 --use_query --fix_bert --train_file data/multiwoz2.0/train_nrs_w-dream.json

output_dir=exp-multiwoz/stage2.0-s1-1.0

python3 code/main-multislot-f.py --do_train --do_eval --num_train_epochs 3 --data_dir data/multiwoz2.0 \
--bert_model bert-base-uncased --do_lower_case --bert_dir $bert_dir --task_name bert-gru-sumbt --nbt rnn --output_dir $output_dir \
--target_slot all --warmup_proportion 0.1 --learning_rate 1e-4 --train_batch_size 8 --eval_batch_size 8 --distance_metric product \
--patience 15 --tf_dir tensorboard --hidden_dim 300 --max_label_length 32 --max_seq_length 64 --max_turn_length 22 \
--use_query --fix_bert --gradient_accumulation_steps 4 \
--pretrain /home/jiaofangkai/dst-multi-woz-2.1/SUMBT/exp-multiwoz/stage2.0-NRS-1.0/pytorch_model.bin

output_dir=exp-multiwoz/stage2.0-s2-1.0

python3 code/main-multislot-f.py --do_train --do_eval --num_train_epochs 50 --data_dir data/multiwoz2.0 \
--bert_model bert-base-uncased --do_lower_case --bert_dir $bert_dir --task_name bert-gru-sumbt --nbt rnn --output_dir $output_dir \
--target_slot all --warmup_proportion 0.1 --learning_rate 5e-5 --train_batch_size 3 --eval_batch_size 4 --distance_metric product \
--patience 15 --tf_dir tensorboard --hidden_dim 300 --max_label_length 32 --max_seq_length 64 --max_turn_length 22 \
--use_query \
--pretrain /home/jiaofangkai/dst-multi-woz-2.1/SUMBT/exp-multiwoz/stage2.0-s1-1.0/pytorch_model.bin

