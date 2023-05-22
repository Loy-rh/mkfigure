import os
from dir_setting import final_df

f_name = os.path.basename(__file__).split('.')[0]

# setting parameter
ylabel = "Accuracy"
Ylabel = "Test accuracy"
# ylabel = 'AUC'
# Ylabel = 'AUC'

dataset = 'cifar100'
arch = 'PreActResNet18'

write_df = ["DivideMix", "Proposed"]
root_dir = './saved'
read_dir = {"DivideMix": '/DivideMix', "Proposed": '/Proposed'}
# root_dir = './saved'
# write_df = ["UPLplus", "Proposed"]
# read_dir = {"UPLplus": "/UPLplus", "Proposed": "/Proposed"}

exp_type_order = ["0.2_sym", "0.5_sym", "0.8_sym", "0.9_sym", "0.4_asym"]
# exp_type_order = ["0.2_sym", "0.5_sym", "0.8_sym", "0.9_sym"]
# exp_type_order = ["0.8_sym", "0.9_sym"]

df = final_df(
    Ylabel, ylabel, dataset, arch, write_df, mode='last_ten_epoch',
    f_name=f_name, root=root_dir, read_dir=read_dir, exp_type_order=exp_type_order
)
