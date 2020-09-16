#!/bin/bash

output_dir='share/exp-multiwoz/data2.1-cls-tf-bi-mask/v2.1'
target_slot='all'
nbt='bi_graph'
bert_dir='/home/jiaofangkai/bert-base-uncased'
bert_name='bert-base-uncased'

# Add layer norm and dropout // remove layer_norm and dropout
# cls_type 0 -> 1 per_eval_steps 500 -> 250 epoch 30 -> 25 -> 20
# Version 1.4: Add relative position embedding in turn GAT
# Version 1.5: It seems version 1.4 run a wrong version. Add initialization and re-run the model.
# Version 1.6: Adopt a low learning rate
# Version 1.6.1: Remove dropout lr -> 1e-4
# Version 2.0: Add dropout / Adjust hyper-parameters
# Version 2.1: Add LayerNorm / warmup-proportion 0.1 -> 0.06

#python -m torch.distributed.launch --nproc_per_node 2 \
#code_transformer/main_torch_1_6_distribute.py --do_train --do_eval --num_train_epochs 40 --data_dir data/multiwoz2.1_5 \
#--bert_model bert-base-uncased --do_lower_case --bert_dir $bert_dir --task_name bert-gru-sumbt --bert_name ${bert_name} \
#--nbt $nbt --output_dir $output_dir --target_slot $target_slot --warmup_proportion 0.06 --learning_rate 1e-4 \
#--train_batch_size 16 --eval_batch_size 1 --distance_metric product --patience 15 \
#--tf_dir share/tensorboard --max_seq_length 64 --max_turn_length 22 \
#--gradient_accumulation_steps 16 \
#--max_label_length 20 --max_slot_length 6  \
#--train_file data/multiwoz2.1_5/train-5-full-value.tsv \
#--dev_file data/multiwoz2.1_5/dev-5-full-value.tsv \
#--test_file data/multiwoz2.1_5/test-5-full-value.tsv \
#--ontology data/multiwoz2.1_5/ontology-full.json \
#--cls_type 1 --value_embedding_type cls --per_eval_steps 250

python \
code_transformer/main_torch_1_6.py --do_train --do_eval --num_train_epochs 40 --data_dir data/multiwoz2.1_5 \
--bert_model bert-base-uncased --do_lower_case --bert_dir $bert_dir --task_name bert-gru-sumbt --bert_name ${bert_name} \
--nbt $nbt --output_dir $output_dir --target_slot $target_slot --warmup_proportion 0.06 --learning_rate 1e-4 \
--train_batch_size 16 --eval_batch_size 1 --distance_metric product --patience 15 \
--tf_dir share/tensorboard --max_seq_length 64 --max_turn_length 22 \
--gradient_accumulation_steps 16 \
--max_label_length 20 --max_slot_length 6  \
--train_file data/multiwoz2.1_5/train-5-full-value.tsv \
--dev_file data/multiwoz2.1_5/dev-5-full-value.tsv \
--test_file data/multiwoz2.1_5/test-5-full-value.tsv \
--ontology data/multiwoz2.1_5/ontology-full.json \
--cls_type 1 --value_embedding_type cls --per_eval_steps 250

