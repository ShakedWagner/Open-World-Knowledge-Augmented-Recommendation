import subprocess

# Training args
data_dir = '/nvcr/stor/fast/afeldman/data/tests/data/ml-1m/proc_data_w_cf'
task_name = 'ctr'
# dataset_name = 'amz'
dataset_name = 'ml-1m'
aug_prefix = 'bert_avg'
augment = True
# augment = False
llm_name = 'Llama-2-7b-chat-hf'

epoch = 20
batch_size = 256
lr = '5e-4'
lr_sched = 'cosine'
weight_decay = 0  #效果不大

model = 'DIN'
# model = 'DIEN'
embed_size = 32
final_mlp = '200,80'
convert_arch = '128,32'
num_cross_layers = 3
dropout = 0.0  #增大会变差

convert_type = 'HEA'
convert_dropout = 0.0
export_num = 2
specific_export_num = 5
dien_gru = 'AIGRU'

save_dir = f'/nvcr/stor/fast/afeldman/data/tests/data/model_llama2_w_cf/{dataset_name}/{task_name}/{model}/WDA_Emb{embed_size}_epoch{epoch}'

# Run the train process

for batch_size in [512]:#256, 512, 2048, 128, 1024,]:
    for lr in ['1e-4']:#['1e-4', '5e-4', '1e-3']:
        for export_num in [2]:
            for specific_export_num in [5]:#[2, 3, 4, 5, 6]:

                print('---------------bs, lr, epoch, export share/spcf, convert arch, gru----------', batch_size,
                      lr, epoch, export_num, specific_export_num, convert_arch, dien_gru, model)
                subprocess.run(['python', '-u', '/home/afeldman/projects/Open-World-Knowledge-Augmented-Recommendation/RS/main_ctr.py',
                                f'--save_dir={save_dir}'
                                f'_bs{batch_size}_lr{lr}_{lr_sched}_cnvt_arch_{convert_arch}_cnvt_type_{convert_type}'
                                f'_eprt_{export_num}_wd{weight_decay}_drop{dropout}' + \
                                f'_hl{final_mlp}_cl{num_cross_layers}_augment_{augment}',
                                f'--data_dir={data_dir}',
                                f'--augment={augment}',
                                f'--aug_prefix={aug_prefix}',
                                f'--task={task_name}',
                                f'--convert_arch={convert_arch}',
                                f'--convert_type={convert_type}',
                                f'--convert_dropout={convert_dropout}',
                                f'--epoch_num={epoch}',
                                f'--batch_size={batch_size}',
                                f'--lr={lr}',
                                f'--lr_sched={lr_sched}',
                                f'--weight_decay={weight_decay}',
                                f'--algo={model}',
                                f'--embed_size={embed_size}',
                                f'--export_num={export_num}',
                                f'--specific_export_num={specific_export_num}',
                                f'--final_mlp_arch={final_mlp}',
                                f'--dropout={dropout}',
                                f'--dien_gru={dien_gru}',
                                f'--llm_name={llm_name}'
                                ])
