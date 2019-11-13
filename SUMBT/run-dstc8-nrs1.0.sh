#!/bin/bash

output_dir=exp-dstc8/nrs-e1-tf1.0
bert_dir='/home/jiaofangkai/'
train_file='/home/jiaofangkai/dstc8-schema-guided-dialogue/train-nrs.json'
dev_file='/home/jiaofangkai/dstc8-schema-guided-dialogue/dev-nrs.json'

python code/main-share-nrs.py --do_train --do_eval --num_train_epochs 1 --data_dir data/multiwoz2.1_5 \
--bert_model bert-base-uncased --do_lower_case --bert_dir $bert_dir --task_name bert-gru-sumbt \
--nbt transformer --output_dir $output_dir --warmup_proportion 0.1 --learning_rate 5e-5 --target_slot all \
--train_batch_size 64 --eval_batch_size 8 --distance_metric product --patience 10 \
--tf_dir tensorboard/dstc/ --max_seq_length 128 --max_turn_length 22 \
--fp16 --fp16_opt_level O2 --gradient_accumulation_steps 8 \
--reduce_layers 2 --num_rnn_layers 4 --max_sample_length 64 \
--train_file $train_file --dev_file $dev_file

