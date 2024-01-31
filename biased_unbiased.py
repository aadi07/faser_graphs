import uproot
import numpy as np
import matplotlib.pyplot as plt
import awkward as ak
import seaborn as sns

sns.set()

relevant_branches = [
    'itParam_align_clus_unbiased_local_res_x',
    'fitParam_align_local_residual_x',
    'itParam_align_clus_unbiased_res_id',
    'fitParam_align_id'
]

def plot(data, sta, lay, mod, param, idparam, biased, r=0.1, draw_layer=False, draw_mod=False):
    plt.xlabel('local_residual_x'+'(mm)', fontsize=20)
    res = ak.flatten(data[param])
    ids = ak.flatten(data[idparam])

    station = (ids // 1000) % 10
    layer = (ids // 100) % 10
    module = (ids // 10) % 10

    if draw_mod:
        res = res[
            ak.where(
                (station==sta) &
                (layer==lay) &
                (module==mod) &
                (res>-r) &
                (res<r)
            )
        ]
    elif draw_layer:
        res = res[
            ak.where(
                (station==sta) &
                (layer==lay) &
                (res>-r) &
                (res<r)
            )
        ]
    else:
        res = res[
            ak.where(
                (station==sta) &
                (res>-r) &
                (res<r)
            )
        ]
    
    label = f'{"biased" if biased else "unbiased"}, µ: {round(np.mean(res), 5)}, σ: {round(np.std(res), 5)}'
    plt.hist(
        res,
        bins=bin,
        histtype='step',
        linewidth=0.7,
        label=label,
        density=True
    )
    plt.ylabel('Number of Events (normalized)', fontsize=20)
    plt.xticks(fontsize=16)
    plt.legend(fontsize=18, loc=1)


with uproot.open('kf_alignment_data.root') as file:
    data = file['trackParam'].arrays(relevant_branches, library='ak')
    for sta in range(4):
        for lay in range(3):
            for mod in range(8):
                fig = plt.figure(figsize=(10, 10))
                bin = np.linspace(-0.1, 0.1, 60)
                plot(data, sta, lay, mod, 'fitParam_align_local_residual_x', 'fitParam_align_id', True, 0.1, True, True)
                plot(data, sta, lay, mod, 'itParam_align_clus_unbiased_local_res_x', 'itParam_align_clus_unbiased_res_id', False, 0.1, True, True)       
                plt.title(f'sta{sta},layer{lay},mod{mod}', fontsize=25)
                fig.savefig(f'station{sta}_layer{lay}_module{mod}_cluster_resx.png')
                plt.close(fig)

    # for sta in range(4):
    #     for lay in range(3):
    #         mod = 0
    #         fig = plt.figure(figsize=(10, 10))
    #         bin = np.linspace(-0.1, 0.1, 60)
    #         plot(data, sta, lay, mod, 'fitParam_align_local_residual_x', 'fitParam_align_id', True, 0.1, True, False)
    #         plot(data, sta, lay, mod, 'itParam_align_clus_unbiased_local_res_x', 'itParam_align_clus_unbiased_res_id', False, 0.1, True, False)       
    #         plt.title(f'sta{sta},layer{lay}', fontsize=25)
    #         fig.savefig(f'figures/station{sta}_layer{lay}_cluster_resx.pdf', format='pdf')
    #         plt.close(fig)
    
    # for sta in range(4):
    #     lay = 0
    #     mod = 0
    #     fig = plt.figure(figsize=(10, 10))
    #     bin = np.linspace(-0.1, 0.1, 60)
    #     plot(data, sta, lay, mod, 'fitParam_align_local_residual_x', 'fitParam_align_id', True, 0.1, False, False)
    #     plot(data, sta, lay, mod, 'itParam_align_clus_unbiased_local_res_x', 'itParam_align_clus_unbiased_res_id', False, 0.1, False, False)       
    #     plt.title(f'sta{sta}', fontsize=25)
    #     fig.savefig(f'figures/station{sta}_cluster_resx.pdf', format='pdf')
    #     plt.close(fig)
