#!/bin/bash

#output_dir=exp-multiwoz/data2.0-share-sa1.0
#output_dir=exp-multiwoz/data2.0-share-sa1.1 # epoch 20 -> 6 gradient accumulation steps 8 -> 1
#output_dir=exp-multiwoz/data2.0-share-sa1.1-g8 # gradient accumulation steps 1 -> 8
#output_dir=exp-multiwoz/data2.0-share-sa1.2  # gradient accumulation steps 8 -> 1 .epoch 6 -> 10
#output_dir=exp-multiwoz/data2.0-share-sa1.3 # epoch 10 -> 6  --sa_add_layer_norm
#output_dir=exp-multiwoz/data2.0-share-sa-tr1.0  # --override_attn
#output_dir=exp-multiwoz/data2.0-share-sa-tr1.2  # max_seq_length = 64 -> 96
#output_dir=exp-multiwoz/data2.0-share-sa-tr1.3  # transpose layer 1 -> 4
#output_dir=exp-multiwoz/data2.0-share-sa-tr1.4  # learning_rate 5e-5 -> 4e-5
output_dir=exp-multiwoz/data2.0-share-sa-tr1.5  # num_train_epochs 6 -> 10
target_slot='all'
nbt='transpose'
bert_dir='/home/jiaofangkai/'

python code/main-multislot-share.py --do_train --do_eval --num_train_epochs 10 --data_dir data/multiwoz2.0 \
--bert_model bert-base-uncased --do_lower_case --bert_dir $bert_dir --task_name bert-gru-sumbt \
--nbt $nbt --output_dir $output_dir --target_slot all --warmup_proportion 0.1 --learning_rate 4e-5 \
--train_batch_size 1 --eval_batch_size 1 --distance_metric product --patience 5 \
--tf_dir tensorboard --max_seq_length 96 --max_turn_length 22 \
--fp16 --fp16_opt_level O2 --gradient_accumulation_steps 1 \
--reduce_layers 0 --max_label_length 17 --max_slot_length 6 \
--override_attn --transpose_layer 4
