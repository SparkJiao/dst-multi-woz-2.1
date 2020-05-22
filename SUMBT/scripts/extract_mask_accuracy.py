import os

p_path = '/home/jiaofangkai/dst-multi-woz-2.1/SUMBT/exp-multiwoz/data2.1-cls-graph2p/v1.4-7.14'

output = []

# for i in range(30):
#     single_res_path = p_path + f'/slot_res_{i}/eval_all_accuracies_pytorch_model.bin.txt'
#
#     output.append(f'====================== slot res {i} ==============================')
#
#     with open(single_res_path, 'r') as f:
#         output.append(f.read())
#
# with open(p_path + '/slot_res_cb.txt', 'w') as f:
#     for x in output:
#         f.write(x)


for i in range(7):
    single_mode_path = p_path + f'/test_mode_{i}/eval_all_accuracies_pytorch_model.bin.txt'

    output.append(f'====================== test mode {i} ==============================\n')

    with open(single_mode_path, 'r') as f:
        output.append(f.read())

    single_mode_path = p_path + f'test_mode_{i}_mask_self/eval_all_accuracies_pytorch_model.bin.txt'

    output.append(f'====================== test mode {i} mask self ==============================\n')

    with open(single_mode_path, 'r') as f:
        output.append(f.read())
