#!/bin/bash

output_dir='/home/jiaofangkai/share/SUMBT/exp-multiwoz/data2.1-cls-tf-flat-copy/v1.2-dop'
#output_dir=exp-multiwoz/data2.1-cls-tf-flat/v1.5-dis
target_slot='all'
nbt='copy'
#bert_dir='/home/admin/workspace/bert-base-uncased'
bert_dir='/home/jiaofangkai/bert-base-uncased'
bert_name='bert-base-uncased'

python \
code_transformer/main_torch_1_6_dop.py --do_train --do_eval --num_train_epochs 20 --data_dir data/multiwoz2.1_5 \
--bert_model bert-base-uncased --do_lower_case --bert_dir $bert_dir --task_name bert-gru-sumbt --bert_name ${bert_name} \
--nbt $nbt --output_dir $output_dir --target_slot $target_slot --warmup_proportion 0.1 --learning_rate 5e-5 \
--train_batch_size 4 --eval_batch_size 1 --distance_metric euclidean --patience 15 \
--tf_dir tensorboard --max_seq_length 64 --max_turn_length 22 \
--gradient_accumulation_steps 4 --attn_head 4 --add_query_attn \
--max_label_length 20 --max_slot_length 6 --num_layers 3 \
--train_file data/multiwoz2.1_5/train-5-full-value.tsv \
--dev_file data/multiwoz2.1_5/dev-5-full-value.tsv \
--test_file data/multiwoz2.1_5/test-5-full-value.tsv \
--ontology data/multiwoz2.1_5/ontology-full.json \
--cls_type 0 --value_embedding_type cls --intermediate_size 1536 --per_eval_steps 1000 --add_interaction --sinusoidal_embeddings \
--op2_lr 1e-4

#CUDA_VISIBLE_DEVICES=0 python \
#code_transformer/main_torch_1_6.py --do_eval --num_train_epochs 30 --data_dir data/multiwoz2.1_5 \
#--bert_model bert-base-uncased --do_lower_case --bert_dir $bert_dir --task_name bert-gru-sumbt --bert_name ${bert_name} \
#--nbt $nbt --output_dir $output_dir --target_slot $target_slot --warmup_proportion 0.1 --learning_rate 1e-4 \
#--train_batch_size 4 --eval_batch_size 1 --distance_metric euclidean --patience 15 \
#--tf_dir tensorboard --max_seq_length 64 --max_turn_length 22 \
#--gradient_accumulation_steps 4 --attn_head 4 --add_query_attn \
#--max_label_length 20 --max_slot_length 6 --num_layers 6 \
#--train_file data/multiwoz2.1_5/train-5-full-value.tsv \
#--dev_file data/multiwoz2.1_5/dev-5-full-value.tsv \
#--test_file data/multiwoz2.1_5/test-5-full-value.tsv \
#--ontology data/multiwoz2.1_5/ontology-full.json \
#--cls_type 0 --value_embedding_type cls --intermediate_size 1536 --per_eval_steps 500 --add_interaction
