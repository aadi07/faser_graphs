import uproot
import numpy as np
import matplotlib.pyplot as plt
import awkward as ak
from lmfit.models import GaussianModel
from math import sqrt
import seaborn as sns

sns.set()

relevant_branches = [
    'itParam_align_clus_unbiased_local_res_x',
    'fitParam_align_local_residual_x',
    'itParam_align_clus_unbiased_res_id',
    'fitParam_align_id',
    'fitParam_chi2',
    'fitParam_nMeasurements',
    'fitParam_pz'
]


biased_data_stats_mod = []
biased_mc_stats_mod = []
unbiased_data_stats_mod = []
unbiased_mc_stats_mod = []
biased_data_stats_lay = []
biased_mc_stats_lay = []
unbiased_data_stats_lay = []
unbiased_mc_stats_lay = []

def get_stats(data, sta, lay, mod, param, idparam, r=0.1, draw_layer=False, draw_mod=False):
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
    
    return np.mean(res), np.std(res)


with uproot.open('kf_alignment_data_v2.root') as file:
    with uproot.open('kf_alignment_mc.root') as mc_file:
        data = file['trackParam'].arrays(relevant_branches, library='ak')
        cut = ak.where((data['fitParam_chi2'] < 100)&(data['fitParam_nMeasurements'] >= 20) & (data['fitParam_pz'] >= 200))
        mc_data = mc_file['trackParam'].arrays(relevant_branches, library='ak')
        mc_cut = ak.where((mc_data['fitParam_chi2'] < 100)&(mc_data['fitParam_nMeasurements'] >= 20) & (mc_data['fitParam_pz'] >= 200))
        data = data[cut]
        mc_data = mc_data[mc_cut]
        for sta in range(4):
            for lay in range(3):
                for mod in range(8):
                    m, s = get_stats(data, sta, lay, mod, 'fitParam_align_local_residual_x', 'fitParam_align_id', 0.1, True, True)
                    biased_data_stats_mod.append((100*sta+10*lay+mod, (m, s)))
                    m, s = get_stats(mc_data, sta, lay, mod, 'fitParam_align_local_residual_x', 'fitParam_align_id', 0.1, True, True)
                    biased_mc_stats_mod.append((100*sta+10*lay+mod, (m, s)))
                    m, s = get_stats(data, sta, lay, mod, 'itParam_align_clus_unbiased_local_res_x', 'itParam_align_clus_unbiased_res_id', 0.1, True, True)
                    unbiased_data_stats_mod.append((100*sta+10*lay+mod, (m, s)))
                    m, s = get_stats(mc_data, sta, lay, mod, 'itParam_align_clus_unbiased_local_res_x', 'itParam_align_clus_unbiased_res_id', 0.1, True, True)
                    unbiased_mc_stats_mod.append((100*sta+10*lay+mod, (m, s)))
                    
        for sta in range(4):
            for lay in range(3):
                mod = 0
                m, s = get_stats(data, sta, lay, mod, 'fitParam_align_local_residual_x', 'fitParam_align_id', 0.1, True, False)
                biased_data_stats_lay.append((10*sta+lay, (m, s)))
                m, s = get_stats(mc_data, sta, lay, mod, 'fitParam_align_local_residual_x', 'fitParam_align_id', 0.1, True, False)
                biased_mc_stats_lay.append((10*sta+lay, (m, s)))
                m, s = get_stats(data, sta, lay, mod, 'itParam_align_clus_unbiased_local_res_x', 'itParam_align_clus_unbiased_res_id', 0.1, True, False)
                unbiased_data_stats_lay.append((10*sta+lay, (m, s)))
                m, s = get_stats(mc_data, sta, lay, mod, 'itParam_align_clus_unbiased_local_res_x', 'itParam_align_clus_unbiased_res_id', 0.1, True, False)
                unbiased_mc_stats_lay.append((10*sta+lay, (m, s)))

fig, (ax1, ax2) = plt.subplots(2, figsize=(30,20))
ax1.scatter(range(len(biased_data_stats_lay)), [i[1][0] for i in biased_data_stats_lay], label="data")
ax1.scatter(range(len(biased_data_stats_lay)), [i[1][0] for i in biased_mc_stats_lay], label="mc")
ax1.tick_params(axis='both', which='major', labelsize=20)
ax1.set_xticks(range(len(biased_data_stats_lay)), [i[0] for i in biased_data_stats_lay], fontsize=20, rotation=90)
ax1.set_title('biased,mean,layer', fontsize=30)
ax1.set_xlabel('Layer ID', fontsize=20)
ax1.set_ylabel('Residual Mean', fontsize=20)
ax1.legend(fontsize=30, loc=1)
ax2.scatter(range(len(biased_data_stats_lay)), [i[1][1] for i in biased_data_stats_lay], label="data")
ax2.scatter(range(len(biased_data_stats_lay)), [i[1][1] for i in biased_mc_stats_lay], label="mc")
ax2.tick_params(axis='both', which='major', labelsize=20)
ax2.set_xticks(range(len(biased_data_stats_lay)), [i[0] for i in biased_data_stats_lay], fontsize=20, rotation=90)
ax2.set_title('biased,stdev,layer', fontsize=30)
ax2.set_xlabel('Layer ID', fontsize=20)
ax2.set_ylabel('Residual STD', fontsize=20)
ax2.legend(fontsize=30, loc=1)
fig.savefig('biased_layer.pdf', format='pdf')
fig, (ax3, ax4) = plt.subplots(2, figsize=(30,20))
ax3.scatter(range(len(unbiased_data_stats_lay)), [i[1][0] for i in unbiased_data_stats_lay], label="data")
ax3.scatter(range(len(unbiased_data_stats_lay)), [i[1][0] for i in unbiased_mc_stats_lay], label="mc")
ax3.tick_params(axis='both', which='major', labelsize=20)
ax3.set_xticks(range(len(unbiased_data_stats_lay)), [i[0] for i in unbiased_data_stats_lay], fontsize=20, rotation=90)
ax3.set_title('unbiased,mean,layer', fontsize=30)
ax3.set_xlabel('Layer ID', fontsize=20)
ax3.set_ylabel('Residual Mean', fontsize=20)
ax3.legend(fontsize=30, loc=1)
ax4.scatter(range(len(unbiased_data_stats_lay)), [i[1][1] for i in unbiased_data_stats_lay], label="data")
ax4.scatter(range(len(unbiased_data_stats_lay)), [i[1][1] for i in unbiased_mc_stats_lay], label="mc")
ax4.tick_params(axis='both', which='major', labelsize=20)
ax4.set_xticks(range(len(unbiased_data_stats_lay)), [i[0] for i in unbiased_data_stats_lay], fontsize=20, rotation=90)
ax4.set_title('unbiased,stdev,layer', fontsize=30)
ax4.set_xlabel('Layer ID', fontsize=20)
ax4.set_ylabel('Residual STD', fontsize=20)
ax4.legend(fontsize=30, loc=1)
fig.savefig('unbiased_layer.pdf', format='pdf')

# fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 10))
fig, (ax1, ax2) = plt.subplots(2, figsize=(30,20))
ax1.scatter(range(len(biased_data_stats_mod)), [i[1][0] for i in biased_data_stats_mod], label="data")
ax1.scatter(range(len(biased_data_stats_mod)), [i[1][0] for i in biased_mc_stats_mod], label="mc")
ax1.tick_params(axis='both', which='major', labelsize=20)
ax1.set_xticks(range(len(biased_data_stats_mod)), [i[0] for i in biased_data_stats_mod], fontsize=20, rotation=90)
ax1.set_title('biased,mean,module', fontsize=30)
ax1.set_xlabel('Module ID', fontsize=20)
ax1.set_ylabel('Residual Mean', fontsize=20)
ax1.legend(fontsize=30, loc=1)
ax2.scatter(range(len(biased_data_stats_mod)), [i[1][1] for i in biased_data_stats_mod], label="data")
ax2.scatter(range(len(biased_data_stats_mod)), [i[1][1] for i in biased_mc_stats_mod], label="mc")
ax2.tick_params(axis='both', which='major', labelsize=20)
ax2.set_xticks(range(len(biased_data_stats_mod)), [i[0] for i in biased_data_stats_mod], fontsize=20, rotation=90)
ax2.set_title('biased,stdev,module', fontsize=30)
ax2.set_xlabel('Module ID', fontsize=20)
ax2.set_ylabel('Residual STD', fontsize=20)
ax2.legend(fontsize=30, loc=1)
fig.savefig('biased_module.pdf', format='pdf')
fig, (ax3, ax4) = plt.subplots(2, figsize=(30,20))
ax3.scatter(range(len(unbiased_data_stats_mod)), [i[1][0] for i in unbiased_data_stats_mod], label="data")
ax3.scatter(range(len(unbiased_data_stats_mod)), [i[1][0] for i in unbiased_mc_stats_mod], label="mc")
ax3.tick_params(axis='both', which='major', labelsize=20)
ax3.set_xticks(range(len(unbiased_data_stats_mod)), [i[0] for i in unbiased_data_stats_mod], fontsize=20, rotation=90)
ax3.set_title('unbiased,mean,module', fontsize=30)
ax3.set_xlabel('Module ID', fontsize=20)
ax3.set_ylabel('Residual Mean', fontsize=20)
ax3.legend(fontsize=30, loc=1)
ax4.scatter(range(len(unbiased_data_stats_mod)), [i[1][1] for i in unbiased_data_stats_mod], label="data")
ax4.scatter(range(len(unbiased_data_stats_mod)), [i[1][1] for i in unbiased_mc_stats_mod], label="mc")
ax4.tick_params(axis='both', which='major', labelsize=20)
ax4.set_xticks(range(len(unbiased_data_stats_mod)), [i[0] for i in unbiased_data_stats_mod], fontsize=20, rotation=90)
ax4.set_title('unbiased,stdev,module', fontsize=30)
ax4.set_xlabel('Module ID', fontsize=20)
ax4.set_ylabel('Residual STD', fontsize=20)
ax4.legend(fontsize=30, loc=1)
fig.savefig('unbiased_module.pdf', format='pdf')