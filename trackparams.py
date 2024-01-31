import uproot
import numpy as np
import matplotlib.pyplot as plt
import awkward as ak

relevant_branches = [
    'fitParam_chi2',
    'fitParam_ndf',
    'fitParam_charge',
    'fitParam_px',
    'fitParam_py',
    'fitParam_pz',
    'fitParam_nMeasurements'
]

def plot(cut_data, cut_mc_data, data, mc_data, plot_title, filename, bins=100):
    fig = plt.figure(figsize=(10, 10))
    
    plt.hist(
        data[cut_data],
        bins=bins,
        histtype='step',
        label=f'data, µ: {round(np.mean(data), 5)}, σ: {round(np.std(data), 5)}',
        density=True
    )
    plt.hist(
        mc_data[cut_mc_data],
        bins=bins,
        histtype='step',
        label=f'mc, µ: {round(np.mean(mc_data), 5)}, σ: {round(np.std(mc_data), 5)}',
        density=True
    )
    plt.xlabel(plot_title, fontsize=20)
    plt.ylabel('Number of Events (normalized)', fontsize=20)
    plt.xticks(fontsize=10)
    plt.legend(fontsize=15,loc=1)
    fig.savefig(f'trackparams/{filename}.pdf', format='pdf')
    plt.close(fig)

with uproot.open('kf_alignment_data_v2.root') as file:
    with uproot.open('kf_alignment_mc.root') as mc_file:
        data = file['trackParam'].arrays(relevant_branches)
        mc_data = mc_file['trackParam'].arrays(relevant_branches)
        cut_data = ak.where((data['fitParam_chi2'] < 100)&(data['fitParam_nMeasurements'] >= 20))
        cut_mc_data = ak.where((mc_data['fitParam_chi2'] < 100)&(mc_data['fitParam_nMeasurements'] >= 20))

        plot(
            cut_data,
            cut_mc_data,
            data['fitParam_chi2']/data['fitParam_ndf'],
            mc_data['fitParam_chi2']/mc_data['fitParam_ndf'],
            'chi2/ndof',
            'chi2_by_ndof',
            bins=np.linspace(0, 3, 100)
        )

        plot(
            cut_data,
            cut_mc_data,
            data['fitParam_charge']/np.sqrt(data['fitParam_px'] ** 2 + data['fitParam_py'] ** 2 + data['fitParam_pz'] ** 2),
            mc_data['fitParam_charge']/np.sqrt(mc_data['fitParam_px'] ** 2 + mc_data['fitParam_py'] ** 2 + mc_data['fitParam_pz'] ** 2),
            'charge/p',
            'charge_by_p'
        )

        plot(
            cut_data,
            cut_mc_data,
            data['fitParam_charge']/np.sqrt(data['fitParam_px'] ** 2 + data['fitParam_py'] ** 2 + data['fitParam_pz'] ** 2),
            mc_data['fitParam_charge']/np.sqrt(mc_data['fitParam_px'] ** 2 + mc_data['fitParam_py'] ** 2 + mc_data['fitParam_pz'] ** 2),
            'charge/pz',
            'charge_by_pz'
        )

        plot(
            cut_data,
            cut_mc_data,
            np.sqrt(data['fitParam_px'] ** 2 + data['fitParam_py'] ** 2 + data['fitParam_pz'] ** 2),
            np.sqrt(mc_data['fitParam_px'] ** 2 + mc_data['fitParam_py'] ** 2 + mc_data['fitParam_pz'] ** 2),
            'p',
            'p'
        )

        plot(
            cut_data,
            cut_mc_data,
            data['fitParam_px'],
            mc_data['fitParam_px'],
            'px',
            'px',
            bins=np.linspace(-5, 5, 100)
        )

        plot(
            cut_data,
            cut_mc_data,
            data['fitParam_py'],
            mc_data['fitParam_py'],
            'py',
            'py',
            bins=np.linspace(-2.5, 2.5, 100)
        )

        plot(
            cut_data,
            cut_mc_data,
            data['fitParam_pz'],
            mc_data['fitParam_pz'],
            'pz',
            'pz',
            bins=np.linspace(0, 2000, 100)
        )