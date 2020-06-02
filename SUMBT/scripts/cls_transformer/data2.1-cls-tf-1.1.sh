#!/bin/bash

output_dir=exp-multiwoz/data2.1-cls-tf/v1.1
target_slot='all'
nbt='transformer'
bert_dir='/home/jiaofangkai/'

python code/main-multislot-share-5-newcls.py --do_train --do_eval --num_train_epochs 40 --data_dir data/multiwoz2.1_5 \
--bert_model bert-base-uncased --do_lower_case --bert_dir $bert_dir --task_name bert-gru-sumbt \
--nbt $nbt --output_dir $output_dir --target_slot $target_slot --warmup_proportion 0.1 --learning_rate 4e-5 \
--train_batch_size 8 --eval_batch_size 1 --distance_metric product --patience 5 \
--tf_dir tensorboard --max_seq_length 90 --max_turn_length 22 \
--fp16 --fp16_opt_level O2 --gradient_accumulation_steps 8 \
--max_label_length 20 --max_slot_length 6 \
--self_attention_type 1 \
--train_file data/multiwoz2.1_5/train-5-full-value.tsv \
--dev_file data/multiwoz2.1_5/dev-5-full-value.tsv \
--test_file data/multiwoz2.1_5/test-5-full-value.tsv \
--ontology data/multiwoz2.1_5/ontology-full.json \
--cls_type 0 --max_loss_scale 256 --value_embedding_type mean --mask_self