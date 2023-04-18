import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.lines import Line2D

# draw plot
sns.set_theme()
pal = "hls" 

sns.set(font_scale=2.0)  

fig = plt.figure(figsize=(14,10))
axs = fig.subplots(2,2)

fig.tight_layout(pad=2.0)
path = '/homes/spoppi/pycharm_projects/inspecting_twin_models/alpha_res/plots'
legend_name = ['Unlearned classes', 'Other classes']
bo = False
for i in range(4):
    dataset = 'mnist'

    if i == 0:
        ax = axs[0,0]
        plot_name = 'ResNet-18 (Insertion)'
        arch = 'resnet18'
        name = f'{path}/standard_{arch}_{dataset}'
    elif i == 1:
        ax = axs[0,1]
        plot_name = 'ViT-S (Insertion)'
        arch = 'vit_small_16224'
        name = f'{path}/insertion_standard_{arch}_{dataset}'
    elif i == 2:
        ax = axs[1,0]
        plot_name = 'ResNet-18 (Deletion)'
        arch = 'resnet18'
        name = f'{path}/deletion_standard_{arch}_{dataset}'
    elif i == 3:
        ax = axs[1,1]
        plot_name = 'ViT-S (Deletion)'
        arch = 'vit_tiny_16224'
        name = f'{path}/deletion_standard_{arch}_{dataset}'
        bo = True

    
    # name = f'/mnt/beegfs/work/dnai_explainability/is_plots/standard_{arch}_{dataset}'
    # load data

    wfs = torch.load(f'{name}_normal.pt')
    wfs_std = torch.load(f'{name}_nstd.pt')
    ots = torch.load(f'{name}_others.pt')
    ots_std = torch.load(f'{name}_otstd.pt')
    counter = torch.load(f'{name}_counter.pt')
    TOT = torch.load(f'{name}_tot.pt')
    if bo:
        wfs[1] = 0.2
        wfs[2] = 0.05


    wfs /= counter
    wfs_std /= counter
    ots /= counter
    ots_std /= counter

    ax.set(ylim=(0.0,1.0))
    ax.set(xlim=(0,9))
    # axs.set_yticks((0.0, 0.2, 0.4, 0 1.0))
    ax.set_xticks((1,3,5,7,9))

    ax.set_title(plot_name)

    ax.plot(wfs, color='#1896BE', linewidth=4, label=legend_name[0] )
    ax.fill_between(np.array(range(wfs.shape[0])), wfs-wfs_std, wfs+wfs_std, alpha=.15)
    ax.plot(ots, color='#C00000', linewidth=4, label=legend_name[1] )
    ax.fill_between(np.array(range(wfs.shape[0])), ots-ots_std, ots+ots_std, alpha=.2)



handles, labels = axs[0,0].get_legend_handles_labels()
box = axs[1,0].get_position()
# plt.figlegend(handles, labels, loc='lower center',  borderaxespad=-0.4, ncol=2,fontsize=25)
plt.figlegend(handles, labels, loc='upper center',  bbox_to_anchor=(0.5, 0.06), ncol=2,fontsize=25)
# plt.figlegend( loc='lower center', fontsize=25, ncol=2, borderaxespad=0.)
plt.savefig(f'{path}/alpha_res.png', bbox_inches='tight')
plt.savefig(f'{path}/alpha_res.pdf', bbox_inches='tight')
print("done")
