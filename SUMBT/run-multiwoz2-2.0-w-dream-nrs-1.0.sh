#!/bin/bash

output_dir=exp-multiwoz/NRS-w-dream1.0
bert_dir='/home/jiaofangkai/'

python3 code/main-nrs.py --do_train --do_eval --num_train_epochs 50 --data_dir data/multiwoz2.0 \
--train_file data/multiwoz2.0/train_nrs_w-dream.json --bert_model bert-base-uncased --do_lower_case \
--bert_dir $bert_dir --task_name bert-gru-sumbt --nbt rnn --output_dir $output_dir --warmup_proportion 0.1 --learning_rate 1e-4 \
--train_batch_size 32 --dev_batch_size 8 --distance_metric product --patience 15 --tf_dir tensorboard --hidden_dim 300 \
--max_query_length 64 --max_seq_length 64 --max_turn_length 22 --fp16 --fp16_opt_level O2 --gradient_accumulation_steps 4 \
--model_id 1 --use_query

