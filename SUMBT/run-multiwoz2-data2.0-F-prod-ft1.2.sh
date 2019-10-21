#!/bin/bash

output_dir=exp-multiwoz/exp-fix2-data2.0-F-prod-ft1.2
target_slot='all'
bert_dir='/home/jiaofangkai/'

# The pre-trained model has not been trained to convergence.

python3 code/main-multislot-f.py --do_train --do_eval --num_train_epochs 50 --data_dir data/multiwoz2.0 \
--bert_model bert-base-uncased --do_lower_case --bert_dir $bert_dir --task_name bert-gru-sumbt --nbt rnn\
 --output_dir $output_dir --target_slot all --warmup_proportion 0.1 --learning_rate 5e-5 \
 --train_batch_size 3 --eval_batch_size 4 --distance_metric product --patience 15 --tf_dir tensorboard \
 --hidden_dim 300 --max_label_length 32 --max_seq_length 64 --max_turn_length 22 --use_query \
 --pretrain /home/jiaofangkai/dst-multi-woz-2.1/SUMBT/exp-multiwoz/NRS-1.5/pytorch_model.bin

