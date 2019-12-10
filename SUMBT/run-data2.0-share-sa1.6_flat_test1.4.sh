#!/bin/bash

#output_dir=exp-multiwoz/data2.0-share-sa1.6-flat_test
#output_dir=exp-multiwoz/data2.0-share-sa1.6-flat_test-1.1  # masked value replace -10000.0 with -40000.0
#output_dir=exp-multiwoz/data2.0-share-sa1.6-flat_test-1.2  # -40000.0 -> -10000.0, fp level O2 -> O1
#output_dir=exp-multiwoz/data2.0-share-sa1.6-flat_test-1.3  # fp level O1 -> O2, num_trian_epochs 6 -> 8 learning_rate 5e-5 -> 4e-5
output_dir=exp-multiwoz/data2.0-share-sa1.6-flat_test-1.4
target_slot='all'
nbt='flat_test'
bert_dir='/home/jiaofangkai/'

python code/main-multislot-share.py --do_train --do_eval --num_train_epochs 8 --data_dir data/multiwoz2.0 \
--bert_model bert-base-uncased --do_lower_case --bert_dir $bert_dir --task_name bert-gru-sumbt \
--nbt $nbt --output_dir $output_dir --target_slot $target_slot --warmup_proportion 0.1 --learning_rate 3e-5 \
--train_batch_size 1 --eval_batch_size 1 --distance_metric product --patience 5 \
--tf_dir tensorboard --max_seq_length 96 --max_turn_length 22 \
--fp16 --fp16_opt_level O2 --gradient_accumulation_steps 1 \
--reduce_layers 0 --max_label_length 17 --max_slot_length 6 \
--override_attn --share_position_weight
