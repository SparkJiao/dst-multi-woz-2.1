#!/bin/bash
# Add position embedding for each dialog turns.


target_slot='all'
nbt='graph2p_tf_pos'
bert_dir='./pretrained-models/'

#lr=1e-5
#lr=2e-5
#lr=3e-5
lr=4e-5
#output_dir=exp-multiwoz/data2.1-cls-graph2-tf/v1.4-7.14.seq108.v2.0.1r${lr}.A100
output_dir=exp-multiwoz/data2.1-cls-graph2-tf/v1.4-7.14.seq108.v2.0.1r${lr}.2080Ti

gpu=$1

CUDA_VISIBLE_DEVICES=${gpu} python code/main-multislot-share-5-newcls.py --do_train --do_eval --num_train_epochs 10 --data_dir data/multiwoz2.1_5 \
--bert_model bert-base-uncased --do_lower_case --bert_dir $bert_dir --task_name bert-gru-sumbt \
--nbt $nbt --output_dir $output_dir --target_slot $target_slot --warmup_proportion 0.1 \
--learning_rate ${lr} \
--train_batch_size 2 --eval_batch_size 1 --distance_metric product --patience 10 \
--tf_dir tensorboard --max_seq_length 108 --max_turn_length 22 \
--fp16 --fp16_opt_level O1 --gradient_accumulation_steps 1 \
--reduce_layers 0 --max_label_length 20 --max_slot_length 6 \
--share_position_weight --self_attention_type 1 \
--train_file data/multiwoz2.1_5/train-5-full-value.tsv \
--dev_file data/multiwoz2.1_5/dev-5-full-value.tsv \
--test_file data/multiwoz2.1_5/test-5-full-value.tsv \
--ontology data/multiwoz2.1_5/ontology-full.json \
--cls_type 0 --extra_nbt --graph_value_sup 1.0 --attn_head 12 --extra_nbt_attn_head 12 --diag_attn_hidden_scale 1.0 \
--max_loss_scale 256 --value_embedding_type mean --sa_add_layer_norm