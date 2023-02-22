import os
from dir_setting import final_df

f_name = os.path.basename(__file__).split('.')[0]

# setting parameter
ylabel = "Accuracy"
Ylabel = "Test accuracy"

dataset = 'cifar10'
arch = 'resnet18'
write_df = ('DivideMix', 'SimCLR+DivideMix')

df = final_df(
    Ylabel, ylabel, dataset, arch, write_df, mode='last_ten_epoch',
    f_name=f_name
)
