import uproot
import numpy as np
import matplotlib.pyplot as plt
import awkward as ak
from lmfit.models import GaussianModel
from math import sqrt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages


sns.set_style("whitegrid")

relevant_branches_new = [
    "itParam_align_clus_unbiased_local_res_x",
    "fitParam_align_local_residual_x",
    "itParam_align_clus_unbiased_res_id",
    "fitParam_align_id",
    "fitParam_chi2",
    "fitParam_ndf",
    "fitParam_charge",
    "fitParam_px",
    "fitParam_py",
    "fitParam_pz",
    "fitParam_x",
    "fitParam_y",
    "fitParam_nMeasurements",
]

relevant_branches_old = [
    "fitParam_align_clus_unbiased_local_res_x",
    "fitParam_align_local_residual_x",
    "fitParam_align_clus_unbiased_res_id",
    "fitParam_align_id",
    "fitParam_chi2",
    "fitParam_ndf",
    "fitParam_charge",
    "Track_Px_atIFT",
    "Track_Py_atIFT",
    "Track_Pz_atIFT",
    "Track_X_atIFT",
    "Track_Y_atIFT",
    "fitParam_nMeasurements",
]

three_sta_path = "kf_alignment_3sta.root"
four_sta_path = "kf_alignment_data_v2.root"
mc_sta_path = "kf_alignment_mc.root"

def gaussfun(data):
    bin=np.linspace(-0.1, 0.1, 26)
    npix = len(data)
    nbins = int(sqrt(npix))
    n, bins = np.histogram(data, bins=nbins, density=True)
    n, bins = np.array(n), np.array(bins)

    # Generate data from bins as a set of points 
    bin_size = abs(bins[1]-bins[0])
    x = np.linspace(start=bins[0]+bin_size/2.0,stop=bins[-2]+bin_size/2.0, num=nbins,endpoint=True)
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

def trackparam_plot(three_data, four_data, mc_data, filetitle, title, xlabel, ylabel, bins=100, calc_median=False):
    fig = plt.figure(figsize=(10, 10))
    if not calc_median:
        sns.histplot(
            three_data,
            element="step",
            bins=bins,
            label="3sta Alignment",
            stat="density",
            fill=False
        )
        sns.histplot(
            four_data,
            element="step",
            bins=bins,
            label="4sta Alignment",
            stat="density",
            fill=False
        )
        sns.histplot(
            mc_data,
            element="step",
            bins=bins,
            label="MC Ideal Geometry",
            stat="density",
            fill=False
        )
    
    else:
        sns.histplot(
            three_data,
            element="step",
            bins=bins,
            label=f"3sta Alignment, µ: {round(np.percentile(three_data, 50), 3)}",
            stat="density",
            fill=False
        )
        sns.histplot(
            four_data,
            element="step",
            bins=bins,
            label=f"4sta Alignment, µ: {round(np.percentile(four_data, 50), 3)}",
            stat="density",
            fill=False
        )
        sns.histplot(
            mc_data,
            element="step",
            bins=bins,
            label=f"MC Ideal Geometry, µ: {round(np.percentile(mc_data, 50), 3)}",
            stat="density",
            fill=False
        )
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    # plt.title(title, fontsize=20)
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel(ylabel, fontsize=15)
    plt.legend(fontsize=15, loc=1)
    fig.savefig(f"figures/{filetitle}", format="pdf")

with uproot.open(four_sta_path) as four_sta_file:
    with uproot.open(mc_sta_path) as mc_file:
        with uproot.open(three_sta_path) as three_sta_file:
            data = four_sta_file["trackParam"].arrays(relevant_branches_new, library="ak")
            data = data[ak.where((data["fitParam_chi2"] < 100)&(data["fitParam_nMeasurements"] >= 20))]
            mc_data = mc_file["trackParam"].arrays(relevant_branches_new, library="ak")
            mc_data = mc_data[ak.where((mc_data["fitParam_chi2"] < 100)&(mc_data["fitParam_nMeasurements"] >= 20))]
            three_sta_data = three_sta_file["trackParam"].arrays(relevant_branches_old, library="ak")
            three_sta_data = three_sta_data[ak.where((three_sta_data["fitParam_chi2"] < 100)&(three_sta_data["fitParam_nMeasurements"] >= 15))]
            fig = plt.figure(figsize=(10, 10))
            trackparam_plot(
                three_sta_data["fitParam_chi2"]/three_sta_data["fitParam_ndf"],
                data["fitParam_chi2"]/data["fitParam_ndf"],
                mc_data["fitParam_chi2"]/mc_data["fitParam_ndf"],
                "chi2_by_ndof.pdf",
                "Chi Squared by Degrees of Freedom",
                "chi2/ndof",
                "Number of Events (normalized)",
                bins=np.linspace(0, 3, 100),
                calc_median=True
            )

            trackparam_plot(
                three_sta_data["fitParam_charge"]/np.sqrt(three_sta_data["Track_Px_atIFT"]**2 + three_sta_data["Track_Py_atIFT"]**2 + three_sta_data["Track_Pz_atIFT"]**2),
                data["fitParam_charge"]/np.sqrt(data["fitParam_px"]**2 + data["fitParam_py"]**2 + data["fitParam_pz"]**2),
                mc_data["fitParam_charge"]/np.sqrt(mc_data["fitParam_px"]**2 + mc_data["fitParam_py"]**2 + mc_data["fitParam_pz"]**2),
                "charge_by_p.pdf",
                "Charge by Absolute Momentum",
                "q/p",
                "Number of Events (normalized)",
            )

            trackparam_plot(
                three_sta_data["fitParam_charge"]/three_sta_data["Track_Pz_atIFT"],
                data["fitParam_charge"]/data["fitParam_pz"],
                mc_data["fitParam_charge"]/mc_data["fitParam_pz"],
                "charge_by_pz.pdf",
                "Charge by Z Momentum",
                "q/pz",
                "Number of Events (normalized)",
            )

            trackparam_plot(
                np.sqrt(three_sta_data["Track_Px_atIFT"]**2 + three_sta_data["Track_Py_atIFT"]**2 + three_sta_data["Track_Pz_atIFT"]**2),
                np.sqrt(data["fitParam_px"]**2 + data["fitParam_py"]**2 + data["fitParam_pz"]**2),
                np.sqrt(mc_data["fitParam_px"]**2 + mc_data["fitParam_py"]**2 + mc_data["fitParam_pz"]**2),
                "p.pdf",
                "Absolute Momentum",
                "p",
                "Number of Events (normalized)",
                bins=np.linspace(0, 2000, 100),
                calc_median=True
            )

            trackparam_plot(
                three_sta_data["Track_Px_atIFT"],
                data["fitParam_px"],
                mc_data["fitParam_px"],
                "px.pdf",
                "X Momentum",
                "px",
                "Number of Events (normalized)",
                bins=np.linspace(-5, 5, 100),
                calc_median=True
            )

            trackparam_plot(
                three_sta_data["Track_Py_atIFT"],
                data["fitParam_py"],
                mc_data["fitParam_py"],
                "py.pdf",
                "X Momentum",
                "py",
                "Number of Events (normalized)",
                bins=np.linspace(-2.5, 2.5, 100),
                calc_median=True
            )

            trackparam_plot(
                three_sta_data["Track_Pz_atIFT"],
                data["fitParam_pz"],
                mc_data["fitParam_pz"],
                "pz.pdf",
                "X Momentum",
                "pz",
                "Number of Events (normalized)",
                bins=np.linspace(0, 2000, 100),
                calc_median=True
            )

            trackparam_plot(
                three_sta_data["Track_Px_atIFT"]/three_sta_data["Track_Pz_atIFT"],
                data["fitParam_px"]/data["fitParam_pz"],
                mc_data["fitParam_px"]/mc_data["fitParam_pz"],
                "px_by_pz.pdf",
                "X Momentum/Z Momentum",
                "px/pz",
                "Number of Events (normalized)",
                bins=np.linspace(-0.03, 0.03, 100)
            )

            trackparam_plot(
                three_sta_data["Track_Px_atIFT"]/three_sta_data["Track_Py_atIFT"],
                data["fitParam_px"]/data["fitParam_py"],
                mc_data["fitParam_px"]/mc_data["fitParam_py"],
                "px_by_py.pdf",
                "X Momentum/Y Momentum",
                "px/py",
                "Number of Events (normalized)",
                bins=np.linspace(-5, 5, 100)
            )

            trackparam_plot(
                three_sta_data["Track_X_atIFT"]/three_sta_data["Track_Y_atIFT"],
                data["fitParam_x"]/data["fitParam_y"],
                mc_data["fitParam_x"]/mc_data["fitParam_y"],
                "x_by_y.pdf",
                "X/Y",
                "x/y",
                "Number of Events (normalized)",
                bins=np.linspace(-10, 10, 100)
            )


def plot(p, data, sta, lay, mod, param, idparam, label, r=0.1, draw_layer=False, draw_mod=False):
    p.set_xlabel("local_residual_x"+"(mm)", fontsize=20)
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
    if label == "mc":
        color = "green"
    elif label == "3sta":
        color = "blue"
    elif label == "4sta":
        color = "orange"
    else:
        color = "black"
    p.hist(
        res,
        bins=bin,
        histtype="step",
        linewidth=2,
        density=True,
        color=color
    )
    p.plot(
        x,
        result.best_fit,
        linewidth=3,
        label=f"{label} µ: {mu}±{mu_err}, σ:{std}±{std_err}",
        color=color
    )
    p.set_ylabel("Number of Events (normalized)", fontsize=20)
    p.set_xticks([0.025*i for i in range(-4, 5)])
    p.legend(fontsize=15, loc=1)


with uproot.open(four_sta_path) as four_sta_file:
    with uproot.open(mc_sta_path) as mc_file:
        with uproot.open(three_sta_path) as three_sta_file:
            data = four_sta_file["trackParam"].arrays(relevant_branches_new, library="ak")
            mc_data = mc_file["trackParam"].arrays(relevant_branches_new, library="ak")
            data = data[ak.where((data["fitParam_chi2"] < 100)&(data["fitParam_nMeasurements"] >= 20) & (data["fitParam_pz"] >= 200))]
            mc_data = mc_data[ak.where((mc_data["fitParam_chi2"] < 100)&(mc_data["fitParam_nMeasurements"] >= 20) & (mc_data["fitParam_pz"] >= 200))]
            three_sta_data = three_sta_file["trackParam"].arrays(relevant_branches_old, library="ak")
            three_sta_data = three_sta_data[ak.where((three_sta_data["fitParam_chi2"] < 100)&(three_sta_data["fitParam_nMeasurements"] >= 15) & (three_sta_data["Track_Pz_atIFT"] >= 200))]
            for sta in range(1,4):
                for lay in range(3):
                    for mod in range(8):
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
                        bin = np.linspace(-0.1, 0.1, 60)
                        plot(ax1, data, sta, lay, mod, "fitParam_align_local_residual_x", "fitParam_align_id", "4sta", 0.1, True, True)
                        plot(ax1, mc_data, sta, lay, mod, "fitParam_align_local_residual_x", "fitParam_align_id", "mc", 0.1, True, True)
                        plot(ax1, three_sta_data, sta, lay, mod, "fitParam_align_local_residual_x", "fitParam_align_id", "3sta", 0.1, True, True)
                        ax1.set_title(f"biased,station{sta},layer{lay},module{mod}", fontsize=25)
                        plot(ax2, data, sta, lay, mod, "itParam_align_clus_unbiased_local_res_x", "itParam_align_clus_unbiased_res_id", "4sta", 0.1, True, True)
                        plot(ax2, mc_data, sta, lay, mod, "itParam_align_clus_unbiased_local_res_x", "itParam_align_clus_unbiased_res_id", "mc", 0.1, True, True)
                        plot(ax2, three_sta_data, sta, lay, mod, "fitParam_align_clus_unbiased_local_res_x", "fitParam_align_clus_unbiased_res_id", "3sta", 0.1, True, True)
                        ax2.set_title(f"unbiased,station{sta},layer{lay},module{mod}", fontsize=25)
                        fig.savefig(f"res_x_v_montecarlo/station{sta}_layer{lay}_module{mod}_cluster_resx.pdf", format="pdf")
                        plt.close(fig)

            for sta in range(1,4):
                for lay in range(3):
                    mod = 0
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
                    bin = np.linspace(-0.1, 0.1, 60)
                    plot(ax1, data, sta, lay, mod, "fitParam_align_local_residual_x", "fitParam_align_id", "4sta", 0.1, True, False)
                    plot(ax1, mc_data, sta, lay, mod, "fitParam_align_local_residual_x", "fitParam_align_id", "mc", 0.1, True, False)
                    plot(ax1, three_sta_data, sta, lay, mod, "fitParam_align_local_residual_x", "fitParam_align_id", "3sta", 0.1, True, False)
                    ax1.set_title(f"biased,station{sta},layer{lay}", fontsize=25)
                    plot(ax2, data, sta, lay, mod, "itParam_align_clus_unbiased_local_res_x", "itParam_align_clus_unbiased_res_id", "4sta", 0.1, True, False)
                    plot(ax2, mc_data, sta, lay, mod, "itParam_align_clus_unbiased_local_res_x", "itParam_align_clus_unbiased_res_id", "mc", 0.1, True, False)
                    plot(ax2, three_sta_data, sta, lay, mod, "fitParam_align_clus_unbiased_local_res_x", "fitParam_align_clus_unbiased_res_id", "3sta", 0.1, True, False)
                    ax2.set_title(f"unbiased,station{sta},layer{lay}", fontsize=25)
                    fig.savefig(f"res_x_v_montecarlo/station{sta}_layer{lay}_cluster_resx.pdf", format="pdf")
                    plt.close(fig)
            
            for sta in range(1,4):
                lay = 0
                mod = 0
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
                bin = np.linspace(-0.1, 0.1, 60)
                plot(ax1, data, sta, lay, mod, "fitParam_align_local_residual_x", "fitParam_align_id", "4sta", 0.1, False, False)
                plot(ax1, mc_data, sta, lay, mod, "fitParam_align_local_residual_x", "fitParam_align_id", "mc", 0.1, False, False)
                plot(ax1, three_sta_data, sta, lay, mod, "fitParam_align_local_residual_x", "fitParam_align_id", "3sta", 0.1, False, False)
                ax1.set_title(f"biased,station{sta}", fontsize=25)
                plot(ax2, data, sta, lay, mod, "itParam_align_clus_unbiased_local_res_x", "itParam_align_clus_unbiased_res_id", "4sta", 0.1, False, False)
                plot(ax2, mc_data, sta, lay, mod, "itParam_align_clus_unbiased_local_res_x", "itParam_align_clus_unbiased_res_id", "mc", 0.1, False, False)
                plot(ax2, three_sta_data, sta, lay, mod, "fitParam_align_clus_unbiased_local_res_x", "fitParam_align_clus_unbiased_res_id", "3sta", 0.1, False, False)
                ax2.set_title(f"unbiased,station{sta}", fontsize=25)
                fig.savefig(f"res_x_v_montecarlo/station{sta}_cluster_resx.pdf", format="pdf")
                plt.close(fig)

