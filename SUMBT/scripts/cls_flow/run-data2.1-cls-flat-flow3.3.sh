#!/bin/bash

output_dir=exp-multiwoz/data2.1-cls-flow/v3.3-s0
target_slot='all'
nbt='flow3'
bert_dir='/home/jiaofangkai/'

#python code/main-multislot-share-5-newcls.py --do_train --do_eval --num_train_epochs 40 --data_dir data/multiwoz2.1_5 \
#--bert_model bert-base-uncased --do_lower_case --bert_dir $bert_dir --task_name bert-gru-sumbt \
#--nbt $nbt --output_dir $output_dir --target_slot $target_slot --warmup_proportion 0.1 --learning_rate 1e-3 \
#--train_batch_size 8 --eval_batch_size 2 --distance_metric product --patience 5 \
#--tf_dir tensorboard --max_seq_length 64 --max_turn_length 22 \
#--fp16 --fp16_opt_level O2 --gradient_accumulation_steps 1 \
#--reduce_layers 0 --max_label_length 20 --max_slot_length 6 \
#--override_attn --share_position_weight --self_attention_type 1 \
#--train_file data/multiwoz2.1_5/train-5-full-value.tsv \
#--dev_file data/multiwoz2.1_5/dev-5-full-value.tsv \
#--test_file data/multiwoz2.1_5/test-5-full-value.tsv \
#--ontology data/multiwoz2.1_5/ontology-full.json \
#--cls_type 1 --flow_layer 0 --use_flow --fix_bert --gradient_accumulation_steps 4

# Stopped at epoch 9


#output_dir2=exp-multiwoz/data2.1-cls-flow/v3.3-s1
#
#python code/main-multislot-share-5-newcls.py --do_train --do_eval --num_train_epochs 5 --data_dir data/multiwoz2.1_5 \
#--bert_model bert-base-uncased --do_lower_case --bert_dir $bert_dir --task_name bert-gru-sumbt \
#--nbt $nbt --output_dir $output_dir2 --target_slot $target_slot --warmup_proportion 0.1 --learning_rate 4e-5 \
#--train_batch_size 1 --eval_batch_size 1 --distance_metric product --patience 5 \
#--tf_dir tensorboard --max_seq_length 64 --max_turn_length 22 \
#--fp16 --fp16_opt_level O2 --gradient_accumulation_steps 1 \
#--reduce_layers 0 --max_label_length 20 --max_slot_length 6 \
#--override_attn --share_position_weight --self_attention_type 1 \
#--train_file data/multiwoz2.1_5/train-5-full-value.tsv \
#--dev_file data/multiwoz2.1_5/dev-5-full-value.tsv \
#--test_file data/multiwoz2.1_5/test-5-full-value.tsv \
#--ontology data/multiwoz2.1_5/ontology-full.json \
#--cls_type 1 --flow_layer 0 --use_flow --dropout 0.3 \
#--pretrain $output_dir/pytorch_model_loss.bin


output_dir2=exp-multiwoz/data2.1-cls-flow/v3.3-s2  # no extra dropout

python code/main-multislot-share-5-newcls.py --do_train --do_eval --num_train_epochs 5 --data_dir data/multiwoz2.1_5 \
--bert_model bert-base-uncased --do_lower_case --bert_dir $bert_dir --task_name bert-gru-sumbt \
--nbt $nbt --output_dir $output_dir2 --target_slot $target_slot --warmup_proportion 0.1 --learning_rate 4e-5 \
--train_batch_size 1 --eval_batch_size 1 --distance_metric product --patience 5 \
--tf_dir tensorboard --max_seq_length 64 --max_turn_length 22 \
--fp16 --fp16_opt_level O2 --gradient_accumulation_steps 1 \
--reduce_layers 0 --max_label_length 20 --max_slot_length 6 \
--override_attn --share_position_weight --self_attention_type 1 \
--train_file data/multiwoz2.1_5/train-5-full-value.tsv \
--dev_file data/multiwoz2.1_5/dev-5-full-value.tsv \
--test_file data/multiwoz2.1_5/test-5-full-value.tsv \
--ontology data/multiwoz2.1_5/ontology-full.json \
--cls_type 1 --flow_layer 0 --use_flow \
--pretrain $output_dir/pytorch_model_loss.bin
