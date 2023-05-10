import os
import time

from make_figure import mk_fig
from dir_setting import final_df

f_name = os.path.basename(__file__).split('.')[0]

# setting parameter
# ylabel = "Accuracy"
# Ylabel = 'Test accuracy'
# ylabel = 'Number_of_labeled_samples'
# Ylabel = 'Rate of labeled samples'
# ylabel = 'AUC'
# Ylabel = 'AUC'
ylabel = 'Precision'
Ylabel = 'Precision'
# ylabel = 'plabel_acc'
# Ylabel = 'Pseudo-Labels Accuracy'
# ylabel = 'plabel_acc_th'
# Ylabel = 'Pseudo-Labels Accuracy'

dataset = 'cifar100'
arch = 'PreActResNet18'
# col_wrap = 2
col_wrap = 4  # 2x2
root_dir = './saved'
# write_df = ["DivideMix", "CRAS"]
# read_dir = {"DivideMix": '/DivideMix', "CRAS": '/CRAS'}
write_df = ["UPL", "UPLplus"]
read_dir = {"UPL": "/UPL", "UPLplus": "/UPLplus"}


exp_type_order = ["0.2_sym", "0.5_sym", "0.8_sym", "0.9_sym"]

# text = ylabel
text = "vs_proposed_" + ylabel

df = final_df(Ylabel, ylabel, dataset, arch, write_df,
              root=root_dir, read_dir=read_dir, exp_type_order=exp_type_order)
print('making df finished')

since = time.time()
mk_fig(
    df, Ylabel, dataset, text, col_wrap=col_wrap, hue_order=write_df, f_name=f_name
)
print(f'elapsed time {time.time()-since:.4f}')


