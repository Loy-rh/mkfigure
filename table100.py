import os
from dir_setting import final_df

f_name = os.path.basename(__file__).split('.')[0]

# setting parameter
ylabel = "Accuracy"
Ylabel = "Test accuracy"

dataset = 'cifar100'
arch = 'PreactResNet18'

# write_df = ["DivideMix", "Proposed"]
# root_dir = '../UPL/saved'
# read_dir = {"DivideMix": '/dividemix', "Proposed": '/UPL'}
# write_df = ["CRAS", "wELR"]
# root_dir = '../saved'
# read_dir = {"CRAS": '/CRAS', "wELR": '/wELR'}
write_df = ["DivideMix", "CRAS"]
root_dir = '../saved'
read_dir = {"DivideMix": '/DivideMix', "CRAS": '/CRAS'}

# exp_type_order = ["0.2_sym", "0.5_sym", "0.8_sym", "0.9_sym", "0.4_asym"]
exp_type_order = ["0.2_sym", "0.5_sym", "0.8_sym", "0.9_sym"]

df = final_df(
    Ylabel, ylabel, dataset, arch, write_df, mode='last_ten_epoch',
    f_name=f_name, root=root_dir, read_dir=read_dir, exp_type_order=exp_type_order
)
