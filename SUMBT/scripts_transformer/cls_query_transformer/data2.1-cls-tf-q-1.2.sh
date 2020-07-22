#!/bin/bash

output_dir_pt=exp-multiwoz/data2.1-cls-tf-q/v1.2-pt
target_slot='all'
nbt='query'
#bert_dir='/home/admin/workspace/bert-base-uncased'
bert_dir='/home/jiaofangkai/bert-base-uncased/'
bert_name='bert-base-uncased'

#python code_transformer/main.py --do_train --do_eval --num_train_epochs 50 --data_dir data/multiwoz2.1_5 \
#--bert_model bert-base-uncased --do_lower_case --bert_dir $bert_dir --task_name bert-gru-sumbt --bert_name ${bert_name} \
#--nbt $nbt --output_dir $output_dir_pt --target_slot $target_slot --warmup_proportion 0.1 --learning_rate 1e-3 \
#--train_batch_size 32 --eval_batch_size 2 --distance_metric product --patience 10 \
#--tf_dir tensorboard --max_seq_length 64 --max_turn_length 22 \
#--fp16 --fp16_opt_level O2 --gradient_accumulation_steps 4 \
#--max_label_length 20 --max_slot_length 6 --num_layers 6 \
#--train_file data/multiwoz2.1_5/train-5-full-value.tsv \
#--dev_file data/multiwoz2.1_5/dev-5-full-value.tsv \
#--test_file data/multiwoz2.1_5/test-5-full-value.tsv \
#--ontology data/multiwoz2.1_5/ontology-full.json \
#--cls_type 0 --max_loss_scale 256 --value_embedding_type mean --intermediate_size 1536 --per_eval_steps 1000 \
#--fix_bert

output_dir=exp-multiwoz/data2.1-cls-tf-q/v1.2

python code_transformer/main.py --do_train --do_eval --num_train_epochs 40 --data_dir data/multiwoz2.1_5 \
--bert_model bert-base-uncased --do_lower_case --bert_dir $bert_dir --task_name bert-gru-sumbt --bert_name ${bert_name} \
--nbt $nbt --output_dir $output_dir --target_slot $target_slot --warmup_proportion 0.1 --learning_rate 1e-4 \
--train_batch_size 32 --eval_batch_size 2 --distance_metric product --patience 10 \
--tf_dir tensorboard --max_seq_length 64 --max_turn_length 22 \
--fp16 --fp16_opt_level O2 --gradient_accumulation_steps 8 \
--max_label_length 20 --max_slot_length 6 --num_layers 6 \
--train_file data/multiwoz2.1_5/train-5-full-value.tsv \
--dev_file data/multiwoz2.1_5/dev-5-full-value.tsv \
--test_file data/multiwoz2.1_5/test-5-full-value.tsv \
--ontology data/multiwoz2.1_5/ontology-full.json \
--cls_type 0 --max_loss_scale 256 --value_embedding_type mean --intermediate_size 1536 --per_eval_steps 500 \
--pretrain $output_dir_pt/pytorch_model_loss.bin