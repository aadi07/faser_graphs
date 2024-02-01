import uproot
import numpy as np
import matplotlib.pyplot as plt
import awkward as ak
from lmfit.models import GaussianModel
from math import sqrt

relevant_branches = [
    'itParam_align_clus_unbiased_local_res_x',
    'fitParam_align_local_residual_x',
    'itParam_align_clus_unbiased_res_id',
    'fitParam_align_id',
    'fitParam_chi2',
    'fitParam_nMeasurements',
    'fitParam_pz'
]

def gaussfun(data):
    bin=np.linspace(-0.1, 0.1, 26)
    npix = len(data)
    nbins = int(sqrt(npix))
    n, bins = np.histogram(data, bins=nbins, density=True)
    n, bins = np.array(n), np.array(bins)

    # Generate data from bins as a set of points 
    bin_size = abs(bins[1]-bins[0])
    x =np.linspace(start=bins[0]+bin_size/2.0,stop=bins[-2]+bin_size/2.0, num=nbins,endpoint=True)
    y = n

    model = GaussianModel()
    params = model.guess(y, x=x)
    result = model.fit(y, params, x=x)
    return bin, x, result

def getmusig(result):
    i = 0
    for _, param in result.params.items():
        i += 1
        if i == 2:
            mu_tmp=round(param.value, 5)
            mu_err_tmp=round(param.stderr, 5)
        elif i == 3:
            std_tmp=round(param.value, 5)
            std_err_tmp=round(param.stderr, 5)
    return mu_tmp, mu_err_tmp, std_tmp, std_err_tmp

def plot(p, data, sta, lay, mod, param, idparam, mc, r=0.1, draw_layer=False, draw_mod=False):
    p.set_xlabel('local_residual_x'+'(mm)', fontsize=20)
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
    
    bin, x, result = gaussfun(res)
    mu, mu_err, std, std_err = getmusig(result)
    p.hist(
        res,
        bins=bin,
        histtype='step',
        linewidth=2,
        label=f'{"mc" if mc else "data"}',
        density=True,
        color=("blue" if mc else "red")
    )
    p.plot(
        x,
        result.best_fit,
        linewidth=3,
        label=f'{"mc_gaussian" if mc else "data_gaussian"} µ: {mu}±{mu_err}, σ:{std}±{std_err}',
        color=("blue" if mc else "red")
    )
    p.set_ylabel('Number of Events (normalized)', fontsize=20)
    p.set_xticks([0.025*i for i in range(-4, 5)])
    p.legend(fontsize=15, loc=1)


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
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
                    bin = np.linspace(-0.1, 0.1, 60)
                    plot(ax1, data, sta, lay, mod, 'fitParam_align_local_residual_x', 'fitParam_align_id', False, 0.1, True, True)
                    plot(ax1, mc_data, sta, lay, mod, 'fitParam_align_local_residual_x', 'fitParam_align_id', True, 0.1, True, True)
                    ax1.set_title(f'biased,station{sta},layer{lay},module{mod}', fontsize=25)
                    plot(ax2, data, sta, lay, mod, 'itParam_align_clus_unbiased_local_res_x', 'itParam_align_clus_unbiased_res_id', False, 0.1, True, True)
                    plot(ax2, mc_data, sta, lay, mod, 'itParam_align_clus_unbiased_local_res_x', 'itParam_align_clus_unbiased_res_id', True, 0.1, True, True)
                    ax2.set_title(f'unbiased,station{sta},layer{lay},module{mod}', fontsize=25)
                    fig.savefig(f'res_x_v_montecarlo/station{sta}_layer{lay}_module{mod}_cluster_resx.pdf', format='pdf')
                    plt.close(fig)

        for sta in range(4):
            for lay in range(3):
                mod = 0
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
                bin = np.linspace(-0.1, 0.1, 60)
                plot(ax1, data, sta, lay, mod, 'fitParam_align_local_residual_x', 'fitParam_align_id', False, 0.1, True, False)
                plot(ax1, mc_data, sta, lay, mod, 'fitParam_align_local_residual_x', 'fitParam_align_id', True, 0.1, True, False)
                ax1.set_title(f'biased,station{sta},layer{lay}', fontsize=25)
                plot(ax2, data, sta, lay, mod, 'itParam_align_clus_unbiased_local_res_x', 'itParam_align_clus_unbiased_res_id', False, 0.1, True, False)
                plot(ax2, mc_data, sta, lay, mod, 'itParam_align_clus_unbiased_local_res_x', 'itParam_align_clus_unbiased_res_id', True, 0.1, True, False)
                ax2.set_title(f'unbiased,station{sta},layer{lay}', fontsize=25)
                fig.savefig(f'res_x_v_montecarlo/station{sta}_layer{lay}_cluster_resx.pdf', format='pdf')
                plt.close(fig)
        
        for sta in range(4):
            lay = 0
            mod = 0
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
            bin = np.linspace(-0.1, 0.1, 60)
            plot(ax1, data, sta, lay, mod, 'fitParam_align_local_residual_x', 'fitParam_align_id', False, 0.1, False, False)
            plot(ax1, mc_data, sta, lay, mod, 'fitParam_align_local_residual_x', 'fitParam_align_id', True, 0.1, False, False)
            ax1.set_title(f'biased,station{sta}', fontsize=25)
            plot(ax2, data, sta, lay, mod, 'itParam_align_clus_unbiased_local_res_x', 'itParam_align_clus_unbiased_res_id', False, 0.1, False, False)
            plot(ax2, mc_data, sta, lay, mod, 'itParam_align_clus_unbiased_local_res_x', 'itParam_align_clus_unbiased_res_id', True, 0.1, False, False)
            ax2.set_title(f'unbiased,station{sta}', fontsize=25)
            fig.savefig(f'res_x_v_montecarlo/station{sta}_cluster_resx.pdf', format='pdf')
            plt.close(fig)
