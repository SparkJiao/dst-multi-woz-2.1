#!/bin/bash

output_dir='share/exp-multiwoz/data2.1-cls-tf-bi-mask/v4.10'
target_slot='all'
nbt='bi_graph_fix'
bert_dir='/home/jiaofangkai/bert-base-uncased'
bert_name='bert-base-uncased'

# Add layer norm and dropout // remove layer_norm and dropout
# cls_type 0 -> 1 per_eval_steps 500 -> 250 epoch 30 -> 25 -> 20
# max_seq_length 64 -> 96
# lr 1e-4 -> 5e-5

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
code_transformer/main_torch_1_6.py --do_train --do_eval --num_train_epochs 50 --data_dir data/multiwoz2.1_5 \
--bert_model bert-base-uncased --do_lower_case --bert_dir $bert_dir --task_name bert-gru-sumbt --bert_name ${bert_name} \
--nbt $nbt --output_dir $output_dir --target_slot $target_slot --warmup_proportion 0.1 --learning_rate 1e-4 \
--train_batch_size 32 --eval_batch_size 1 --distance_metric product --patience 15 \
--tf_dir share/tensorboard --max_seq_length 96 --max_turn_length 22 \
--gradient_accumulation_steps 32 \
--max_label_length 20 --max_slot_length 6  \
--train_file data/multiwoz2.1_5/train-5-full-value.tsv \
--dev_file data/multiwoz2.1_5/dev-5-full-value.tsv \
--test_file data/multiwoz2.1_5/test-5-full-value.tsv \
--ontology data/multiwoz2.1_5/ontology-full.json \
--cls_type 1 --value_embedding_type cls --per_eval_steps 250 --pos_emb_per_layer --cls_n_head 12 --cls_d_head 64

