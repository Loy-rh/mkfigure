import os
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
# from figure import result_sum
sns.set(font_scale=2.5)
# sns.set_context("notebook", font_scale=1.5)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'Times New Roman'

dt_d = datetime.now().strftime('%Y%m_%d')
dt_s = datetime.now().strftime('%H%M%S')


def mk_fig(
        df, Ylabel, dataset, text, col_wrap=4, ylim=None, hue_order=('standard',  'finetune'),
        palette=None, f_name=None, legend=False, epoch=300
):

    if col_wrap == 1:
        # common setting
        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel(Ylabel)
        # figsize=(4, 4)
    else:
        print(df.tail(10))
        plt.grid(True)

        kws = dict(linewidth=.8)
        g = sns.FacetGrid(
            df, col='Noise Type', hue='Method', hue_order=hue_order,
            palette=palette, col_wrap=col_wrap, height=5
        )
        g = g.map(sns.lineplot, 'Epoch', Ylabel, **kws).add_legend()
        g.set_titles("{col_name}")
        g.fig.subplots_adjust(wspace=0.1)
        g.set(ylim=ylim)

        # the process required to create a legend that is similar to the original paper
        lg = g.fig.legends[0]
        handles, labels = g.fig.axes[0].get_legend_handles_labels()
        lg.remove()

        if col_wrap == 4:
            if legend:
                # g.fig.axes[1].legend(handles, labels, bbox_to_anchor=(1, 1), loc='lower center',
                #                      ncol=len(hue_order), frameon=True, borderaxespad=3)
                g.fig.axes[1].legend(handles, labels, bbox_to_anchor=(1, 1), loc='lower center',
                                     ncol=len(hue_order), frameon=True, borderaxespad=3)
            # g.set(xticks=[0,50,100,150,200])
            if epoch == 300:
                g.set(xticks=[0, 50, 100, 150, 200, 250, 300])
            else:
                g.set(xticks=[0, 20, 40, 60, 80, 100])
            # if dataset == 'cifar100':
            #     g.set(yticks=[0, 10, 20, 30, 40, 50, 60, 70, 80])
            # figsize = (16, 4)
        elif col_wrap == 2:
            if legend:
                g.fig.axes[0].legend(handles, labels, bbox_to_anchor=(1, 1), loc='lower center',
                                     ncol=len(hue_order), frameon=True, borderaxespad=3)
            if epoch == 300:
                g.set(xticks=[0, 50, 100, 150, 200, 250, 300])
            else:
                g.set(xticks=[0, 20, 40, 60, 80, 100])
            # g.set(yticks=[10, 20, 30, 40, 50, 60, 70])
            # figsize = (12, 8)
        else:
            raise NotImplementedError

    folder = f"./result/{f_name}/{dt_d + dt_s}"
    if not os.path.exists(folder):
        os.system(f"mkdir -p {folder}")
    fig = plt.gcf()
    # fig.set_size_inches(figsize)

    # save figure
    fig.savefig(f"{folder}/figure{dataset}_{text}.png", bbox_inches='tight', pad_inches=0.1)
    pp = PdfPages(f"{folder}/figure{dataset}_{text}.pdf")
    print(f"save figure to {folder}/figure{dataset}_{text}.png")
    pp.savefig(fig, bbox_inches='tight', pad_inches=0.1)
    pp.close()
    plt.clf()


def mk_fig_slide(
        df, Ylabel, dataset, text, col_wrap=4, ylim=(25, 65), hue_order=('standard', 'finetune'),
        palette=None, f_name=None
):

    print(df.tail(10))
    plt.grid(True)

    g = sns.FacetGrid(
        df, col='Noise Type', hue="Method", hue_order=hue_order, palette=palette, col_wrap=col_wrap
    )
    g = g.map(sns.lineplot, 'Epoch', Ylabel).add_legend()
    g.set_titles("{col_name}")
    g.fig.subplots_adjust(wspace=0.1)

    g.set(ylim=ylim)

    folder = f"./result/{f_name}/{dt_d + dt_s}"

    if not os.path.exists(folder):
        os.system(f"mkdir -p {folder}")
    fig = plt.gcf()

    g.fig.subplots_adjust(top=0.8)
    fig.set_size_inches(10, 8 / 1.5)

    # save figure
    fig.savefig(f"{folder}/figure{dataset}_{text}.png", bbox_inches='tight', pad_inches=0.1)
    pp = PdfPages(f"{folder}/figure{dataset}_{text}.pdf")
    print(f"save figure to {folder}/figure{dataset}_{text}.png")
    pp.savefig(fig, bbox_inches='tight', pad_inches=0.1)
    pp.close()
    plt.clf()
