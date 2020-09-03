#!/bin/bash

output_dir='share/exp-multiwoz/data2.1-cls-tf-bi-mask/v1.0'
target_slot='all'
nbt='bi_graph'
bert_dir='/home/jiaofangkai/bert-base-uncased'
bert_name='bert-base-uncased'

#python -m torch.distributed.launch --nproc_per_node 2 \
#code_transformer/main_torch_1_6_distribute.py --do_train --do_eval --num_train_epochs 15 --data_dir data/multiwoz2.1_5 \
#--bert_model bert-base-uncased --do_lower_case --bert_dir $bert_dir --task_name bert-gru-sumbt --bert_name ${bert_name} \
#--nbt $nbt --output_dir $output_dir --target_slot $target_slot --warmup_proportion 0.1 --learning_rate 1e-4 \
#--train_batch_size 16 --eval_batch_size 1 --distance_metric euclidean --patience 15 \
#--tf_dir share/tensorboard --max_seq_length 64 --max_turn_length 22 \
#--gradient_accumulation_steps 16 \
#--max_label_length 20 --max_slot_length 6  \
#--train_file data/multiwoz2.1_5/train-5-full-value.tsv \
#--dev_file data/multiwoz2.1_5/dev-5-full-value.tsv \
#--test_file data/multiwoz2.1_5/test-5-full-value.tsv \
#--ontology data/multiwoz2.1_5/ontology-full.json \
#--cls_type 0 --value_embedding_type cls --per_eval_steps 500

python \
code_transformer/main_torch_1_6.py --do_train --do_eval --num_train_epochs 30 --data_dir data/multiwoz2.1_5 \
--bert_model bert-base-uncased --do_lower_case --bert_dir $bert_dir --task_name bert-gru-sumbt --bert_name ${bert_name} \
--nbt $nbt --output_dir $output_dir --target_slot $target_slot --warmup_proportion 0.1 --learning_rate 1e-4 \
--train_batch_size 16 --eval_batch_size 1 --distance_metric euclidean --patience 15 \
--tf_dir share/tensorboard --max_seq_length 64 --max_turn_length 22 \
--gradient_accumulation_steps 16 \
--max_label_length 20 --max_slot_length 6  \
--train_file data/multiwoz2.1_5/dev-5-full-value.tsv \
--dev_file data/multiwoz2.1_5/dev-5-full-value.tsv \
--test_file data/multiwoz2.1_5/test-5-full-value.tsv \
--ontology data/multiwoz2.1_5/ontology-full.json \
--cls_type 0 --value_embedding_type cls --per_eval_steps 500 --efficient

#CUDA_VISIBLE_DEVICES=0 python \
#code_transformer/main_torch_1_6.py --do_eval --num_train_epochs 30 --data_dir data/multiwoz2.1_5 \
#--bert_model bert-base-uncased --do_lower_case --bert_dir $bert_dir --task_name bert-gru-sumbt --bert_name ${bert_name} \
#--nbt $nbt --output_dir $output_dir --target_slot $target_slot --warmup_proportion 0.1 --learning_rate 1e-4 \
#--train_batch_size 4 --eval_batch_size 1 --distance_metric euclidean --patience 15 \
#--tf_dir share/tensorboard --max_seq_length 64 --max_turn_length 22 \
#--gradient_accumulation_steps 4 --attn_head 4 --add_query_attn \
#--max_label_length 20 --max_slot_length 6 --num_layers 6 \
#--train_file data/multiwoz2.1_5/train-5-full-value.tsv \
#--dev_file data/multiwoz2.1_5/dev-5-full-value.tsv \
#--test_file data/multiwoz2.1_5/test-5-full-value.tsv \
#--ontology data/multiwoz2.1_5/ontology-full.json \
#--cls_type 0 --value_embedding_type cls --intermediate_size 768 --per_eval_steps 500
