#!/bin/bash

#output_dir=exp-multiwoz/data2.1-share-cache-type-prop/v1.1  # eval accroding global step; num_tran_epochs 6->5, lr 5e-5 -> 1e-4
output_dir=exp-multiwoz/data2.1-share-cache-type-prop/v1.2  # distance metric product -> cosine
target_slot='all'
nbt='sa_cache_type_prop'
bert_dir='/home/jiaofangkai/'

python code/main-multislot-share-5-step.py --do_train --do_eval --num_train_epochs 5 --data_dir data/multiwoz2.1_5 \
--bert_model bert-base-uncased --do_lower_case --bert_dir $bert_dir --task_name bert-gru-sumbt \
--nbt $nbt --output_dir $output_dir --target_slot $target_slot --warmup_proportion 0.1 --learning_rate 1e-4 \
--train_batch_size 1 --eval_batch_size 1 --distance_metric cosine --patience 5 \
--tf_dir tensorboard --max_seq_length 64 --max_turn_length 22 \
--fp16 --fp16_opt_level O1 --gradient_accumulation_steps 1 \
--reduce_layers 0 --max_label_length 20 --max_slot_length 6 \
--override_attn --share_position_weight --self_attention_type 2 \
--train_file data/multiwoz2.1_5/train-5-full-value.tsv \
--dev_file data/multiwoz2.1_5/dev-5-full-value.tsv \
--test_file data/multiwoz2.1_5/test-5-full-value.tsv \
--ontology data/multiwoz2.1_5/ontology-full.json \
--per_eval_steps 3000
