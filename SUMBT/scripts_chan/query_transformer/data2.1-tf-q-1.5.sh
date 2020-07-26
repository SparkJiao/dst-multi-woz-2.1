#!/bin/bash

output_dir=exp-multiwoz/data2.1-tf-q/v1.5  # loss is averaged over dialogue size
target_slot='all'
nbt='query'
bert_dir='/home/admin/workspace/bert-base-uncased'
# bert_dir='/home/jiaofangkai/bert-base-uncased/'
bert_name='bert-base-uncased'

python code_chan/main_update_fine.py --do_train --do_eval --num_train_epochs 100 --data_dir data/multiwoz2.1_5 \
--bert_model bert-base-uncased --do_lower_case --bert_dir $bert_dir --task_name bert-gru-sumbt --bert_name ${bert_name} \
--nbt $nbt --output_dir $output_dir --target_slot $target_slot --warmup_proportion 0.1 --learning_rate 5e-5 \
--train_batch_size 4 --eval_batch_size 4 --distance_metric euclidean --patience 20 \
--tf_dir tensorboard --max_seq_length 64 --max_turn_length 22 \
--fp16 --fp16_opt_level O2 --gradient_accumulation_steps 1 \
--max_label_length 20 --max_slot_length 6 --num_layers 6 --attn_head 4 \
--train_file data/multiwoz2.1_5_update/train-5-full-value-update.tsv \
--dev_file data/multiwoz2.1_5_update/dev-5-full-value-update.tsv \
--test_file data/multiwoz2.1_5_update/test-5-full-value-update.tsv \
--ontology data/multiwoz2.1_5_update/ontology-full.json \
--max_loss_scale 64 --value_embedding_type cls --intermediate_size 300 --per_eval_steps 2000