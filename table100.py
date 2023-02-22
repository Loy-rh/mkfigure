import os
from dir_setting import final_df

f_name = os.path.basename(__file__).split('.')[0]

# setting parameter
ylabel = "Accuracy"
Ylabel = "Test accuracy"

dataset = 'cifar100'
# arch = 'resnet18'
arch = 'PreactResNet18'
option = 'best'
# write_df = ('DivideMix', 'SimCLR+DivideMix')
write_df = ('DivideMix', 'Proposed')

df = final_df(
    Ylabel, ylabel, dataset, arch, write_df, mode='last_ten_epoch', f_name=f_name
    # Ylabel, ylabel, dataset, arch, write_df, mode='last_ten_epoch', opt=option, f_name=f_name
)
