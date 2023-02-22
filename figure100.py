import os
import time

from make_figure import mk_fig
from dir_setting import final_df

f_name = os.path.basename(__file__).split('.')[0]

# setting parameter
# ylabel = "Accuracy"
# Ylabel = 'Test accuracy'
# ylabel = 'Number_of_labeled_samples'
# Ylabel = 'Rate of unlabeled samples'
ylabel = 'AUC'
Ylabel = 'AUC'

dataset = 'cifar100'
# arch = 'resnet18'
arch = 'PreactResNet18'
# col_wrap = 2
col_wrap = 4  # 2x2
# write_df = ('DivideMix', 'SimCLR+DivideMix')
write_df = ('DivideMix', "Proposed")
# write_df = ('SimCLR+DivideMix', 'Proposed', 'Proposed(kl)')
# write_df = ('DivideMix', 'DivideMix(-Lu)', 'SimCLR+DivideMix', 'SimCLR+DivideMix(-Lu)')

# text = ylabel
text = "vs_proposed_" + ylabel

df = final_df(Ylabel, ylabel, dataset, arch, write_df)
print('making df finished')

since = time.time()
mk_fig(
    df, Ylabel, dataset, text, col_wrap=col_wrap, hue_order=write_df,
    f_name=f_name
)
print(f'elapsed time {time.time()-since:.4f}')


