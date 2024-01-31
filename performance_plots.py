import uproot 
import numpy as np
import matplotlib.pyplot as plt
import awkward as ak
from lmfit.models import GaussianModel
from math import sqrt
from matplotlib.backends.backend_pdf import PdfPages

branches=[
  "fitParam_chi2",
  "fitParam_ndf",
  "fitParam_nMeasurements",
  "fitParam_align_ift_id",
  "fitParam_align_id",
  "fitParam_align_local_residual_x",
  "fitParam_align_ift_local_residual_x", 
  "fitParam_align_ift_local_residual_y"]
#   "fitParam_align_ift_local_derivation_x_x",
#   "fitParam_align_ift_local_derivation_x_y",
#   "fitParam_align_ift_local_derivation_x_z",
#   "fitParam_align_ift_local_derivation_x_rx", 
#   "fitParam_align_ift_local_derivation_x_ry", 
#   "fitParam_align_ift_local_derivation_x_rz", 
#   "fitParam_align_ift_local_derivation_y_x",
#   "fitParam_align_ift_local_derivation_y_y",
#   "fitParam_align_ift_local_derivation_y_z",
#   "fitParam_align_ift_local_derivation_y_rx", 
#   "fitParam_align_ift_local_derivation_y_ry", 
#   "fitParam_align_ift_local_derivation_y_rz", 
#   "fitParam_align_ift_global_derivation_x_x", 
#   "fitParam_align_ift_global_derivation_x_y", 
#   "fitParam_align_ift_global_derivation_x_z", 
#   "fitParam_align_ift_global_derivation_x_rx",
#   "fitParam_align_ift_global_derivation_x_ry",
#   "fitParam_align_ift_global_derivation_x_rz",
#   "fitParam_align_ift_global_derivation_y_x", 
#   "fitParam_align_ift_global_derivation_y_y", 
#   "fitParam_align_ift_global_derivation_y_z", 
#   "fitParam_align_ift_global_derivation_y_rx",
#   "fitParam_align_ift_global_derivation_y_ry",
#   "fitParam_align_ift_global_derivation_y_rz",
#   "fitParam_align_ift_local_derivation_x_par_x", 
#   "fitParam_align_ift_local_derivation_x_par_y", 
#   "fitParam_align_ift_local_derivation_x_par_theta",
#   "fitParam_align_ift_local_derivation_x_par_phi", 
#   "fitParam_align_ift_local_derivation_x_par_qop", 
#   "fitParam_align_ift_local_derivation_y_par_x", 
#   "fitParam_align_ift_local_derivation_y_par_y", 
#   "fitParam_align_ift_local_derivation_y_par_theta",
#   "fitParam_align_ift_local_derivation_y_par_phi", 
#   "fitParam_align_ift_local_derivation_y_par_qop"]

def ReadBranches(f, branch):
    tree = f['trackParam']
    ts = tree.arrays(branch, library='ak')  
    return ts

def cut(data, Station, Layer, mod, draw_mod=False, draw_layer=False, param='fitParam_align_local_residual_x', id_param='fitParam_align_ift_id'):
    # cut = ak.where((data['fitParam_chi2'] < 200)&(data['fitParam_nMeasurements'] > 6))
    res_x = ak.flatten(data[param])
    id = ak.flatten(data[id_param])

    station=(id//1000)%10
    layer=(id//100)%10
    module=(id//10)%10
    # print(station, layer, module)
    if draw_mod:
        return res_x[ak.where((station==Station)&(layer==Layer)&(module==mod))]
    elif draw_layer:
        return res_x[ak.where((station==Station)&(layer==Layer))]
    else:
        return res_x[ak.where(station==Station)]

def gaussfun(data):
    bin=np.linspace(-0.1, 0.1, 26)
    npix = len(data)
    nbins = int(sqrt(npix))
    #data = np.random.standard_normal(npix)
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
    for name, param in result.params.items():
        i += 1
        if i == 2:
            mu_tmp=round(param.value, 5)
            mu_err_tmp=round(param.stderr, 5)
        elif i == 3:
            std_tmp=round(param.value, 5)
            std_err_tmp=round(param.stderr, 5)
    return mu_tmp, mu_err_tmp, std_tmp, std_err_tmp


def plot(TS, station, layer, mod, label, draw_mod=False, draw_layer=False, param='fitParam_align_local_residual_x', id='fitParam_align_ift_id', range=0.1):
    # Generate data from random Guassian distribution
    data = cut(TS, station, layer, mod, draw_mod, draw_layer, param, id)
    data = data[ak.where((data>-range)&(data<range))]
    bin=np.linspace(-range, range, 60)
    # bin, x, result = gaussfun(data)
    # mu, mu_err, std, std_err = getmusig(result)
    plt.xlabel(param+'(mm)', fontsize=20)
    plt.hist(data, bins=bin, histtype='step', linewidth=0.7, label=f'µ: {round(np.mean(data), 5)}, σ: {round(np.std(data), 5)}')
    # plt.plot(x, result.best_fit, linewidth=2, label=f'{label} µ: {mu}±{mu_err}, σ:{std}±{std_err}')    
    plt.ylabel('Number of Events (normalized)', fontsize=20)
    plt.xticks(fontsize=16)
    plt.legend(fontsize=18)
    # plt.xlim(-0.06, 0.06)
    if draw_layer == True:
        plt.title(f'sta{station},layer{layer}', fontsize=25)
    elif draw_mod == True:
        plt.title(f'sta{station},layer{layer},mod{mod}', fontsize=25)
    else:
        plt.title(f'sta{station}', fontsize=25)


f0 = uproot.open('/data/agarabag/alignment_output/output_data_mod_2_param_ke/0/0_combined_root.root')
data = ReadBranches(f0, branches)


###plot chi2
fig_chi2 = plt.figure(figsize=(10, 10))
plt.hist(data['fitParam_chi2'], bins=100, histtype='step')
plt.xlabel('chi2', fontsize=20)
plt.ylabel('Number of Events', fontsize=20)
plt.xticks(fontsize=20)
fig_chi2.savefig('../docs/IFT_2022_alpha/pre/chi2/chi2.png')
plt.close(fig_chi2)

fig_chi2_ndf = plt.figure(figsize=(10, 10))
plt.hist(data['fitParam_chi2']/data['fitParam_ndf'], bins=100, histtype='step', label="µ:"+str(round(np.percentile(data['fitParam_chi2']/data["fitParam_ndf"], 50), 3)))
plt.xlabel('chi2/ndf', fontsize=20)
plt.ylabel('Number of Events', fontsize=20)
plt.legend(fontsize=18)
plt.xticks(fontsize=20)
fig_chi2_ndf.savefig('../docs/IFT_2022_alpha/pre/chi2/chi2_ndof.png')
plt.close(fig_chi2_ndf)

for sta in range(0, 4):
    mod=0
    lay=0
    fig = plt.figure(figsize=(10, 10))
    plot(data, sta, lay, mod, '', draw_mod=False, draw_layer=False, param='fitParam_align_local_residual_x', id='fitParam_align_id', range=0.1)
    fig.savefig(f'../docs/IFT_2022_alpha/pre/station/station{sta}_cluster_resx.png')
    plt.close(fig)

for sta in range(0, 4):
    for lay in range(3):
        mod=0
        fig = plt.figure(figsize=(10, 10))
        plot(data, sta, lay, mod, '', draw_mod=False, draw_layer=True, param='fitParam_align_local_residual_x', id='fitParam_align_id', range=0.1)
        fig.savefig(f'../docs/IFT_2022_alpha/pre/layer/station{sta}_layer{lay}_cluster_resx.png')
        plt.close(fig)

for sta in range(0, 4):
    for lay in range(3):
        for mod in range(0, 8):
            fig = plt.figure(figsize=(10, 10))
            plot(data, sta, lay, mod, '', draw_mod=True, draw_layer=False, param='fitParam_align_local_residual_x', id='fitParam_align_id', range=0.1)
            fig.savefig(f'../docs/IFT_2022_alpha/pre/module/station{sta}_layer{lay}_module{mod}_cluster_resx.png')
            plt.close(fig)

#SP plots
fig = plt.figure(figsize=(10, 10))
plot(data, 0, 0, 0, '', draw_mod=False, draw_layer=False, param='fitParam_align_ift_local_residual_x', id='fitParam_align_ift_id', range=0.1)
fig.savefig('../docs/IFT_2022_alpha/pre/station/station0_spacepoint_resx.png')
plt.close(fig)

for lay in range(3):
    mod=0
    fig = plt.figure(figsize=(10, 10))
    plot(data, 0, lay, mod, '', draw_mod=False, draw_layer=True, param='fitParam_align_ift_local_residual_x', id='fitParam_align_ift_id', range=0.1)
    fig.savefig(f'../docs/IFT_2022_alpha/pre/layer/station0_layer{lay}_spacepoint_resx.png')
    plt.close(fig)

for lay in range(3):
    for mod in range(0, 8):
        fig = plt.figure(figsize=(10, 10))
        plot(data, 0, lay, mod, '', draw_mod=True, draw_layer=False, param='fitParam_align_ift_local_residual_x', id='fitParam_align_ift_id', range=0.1)
        fig.savefig(f'../docs/IFT_2022_alpha/pre/module/station0_layer{lay}_module{mod}_spacepoint_resx.png')
        plt.close(fig)

figy = plt.figure(figsize=(10, 10))
plot(data, 0, 0, 0, '', draw_mod=False, draw_layer=False, param='fitParam_align_ift_local_residual_y', id='fitParam_align_ift_id', range=3)
figy.savefig('../docs/IFT_2022_alpha/pre/station/station0_spacepoint_resy.png')
plt.close(figy)

for lay in range(3):
    mod=0
    fig = plt.figure(figsize=(10, 10))
    plot(data, 0, lay, mod, '', draw_mod=False, draw_layer=True, param='fitParam_align_ift_local_residual_y', id='fitParam_align_ift_id', range=3)
    fig.savefig(f'../docs/IFT_2022_alpha/pre/layer/station0_layer{lay}_spacepoint_resy.png')
    plt.close(fig)

for lay in range(3):
    for mod in range(0, 8):
        fig = plt.figure(figsize=(10, 10))
        plot(data, 0, lay, mod, '', draw_mod=True, draw_layer=False, param='fitParam_align_ift_local_residual_y', id='fitParam_align_ift_id', range=3)
        fig.savefig(f'../docs/IFT_2022_alpha/pre/module/station0_layer{lay}_module{mod}_spacepoint_resy.png')
        plt.close(fig)


# filenames = ['/home/agarabag/millepede/target/algn_S1L0_v2_iter1.res', 
#              '/home/agarabag/millepede/target/algn_S1L0_v2_iter2.res',
#              '/home/agarabag/millepede/target/algn_S1L0_v2_iter3.res',
#              '/home/agarabag/millepede/target/algn_S1L0_v2_iter4.res',
#              '/home/agarabag/millepede/target/algn_S1L0_v2_iter5.res',
#              '/home/agarabag/millepede/target/algn_S1L0_v2_iter6.res']

# filenames = ['/home/agarabag/millepede/target/ift_clus_ini.res', 
#              '/home/agarabag/millepede/target/ift_clus_iter1.res',
#              '/home/agarabag/millepede/target/ift_clus_iter2.res',
#              '/home/agarabag/millepede/target/ift_clus_iter3.res', 
#              '/home/agarabag/millepede/target/ift_clus_iter4.res',
#              '/home/agarabag/millepede/target/ift_clus_iter5.res', 
#              '/home/agarabag/millepede/target/ift_clus_iter6.res',
#              '/home/agarabag/millepede/target/ift_clus_iter7.res']

# filenames = ['/home/agarabag/millepede/target/millepede_0.res',
#              '/home/agarabag/millepede/target/millepede_1.res',
#              '/home/agarabag/millepede/target/millepede_2.res',
#              '/home/agarabag/millepede/target/millepede_3.res',
#              '/home/agarabag/millepede/target/millepede_4.res',
#              '/home/agarabag/millepede/target/millepede_5.res'
#              ]

# x_values = []
# y_values = []
# rx_values = []
# ry_values = []
# rz_values = []
# x_err = []
# y_err = []
# rx_err = []
# ry_err = []
# rz_err = []

# l2_x_values = []
# l2_y_values = []
# l2_rx_values = []
# l2_ry_values = []
# l2_rz_values = []
# l2_x_err = []
# l2_y_err = []
# l2_rx_err = []
# l2_ry_err = []
# l2_rz_err = []

# l3_x_values = []
# l3_y_values = []
# l3_rx_values = []
# l3_ry_values = []
# l3_rz_values = []
# l3_x_err = []
# l3_y_err = []
# l3_rx_err = []
# l3_ry_err = []
# l3_rz_err = []

# for filename in filenames:
#     with open(filename, 'r') as file:
#         lines = file.readlines()
#     for idx, line in enumerate(lines):
#         columns = line.split()
#         if idx == 1:
#             x_values.append(float(columns[3]))
#             x_err.append(float(columns[4]))
#         elif idx == 2:
#             y_values.append(float(columns[3]))
#             y_err.append(float(columns[4]))
#         elif idx == 5:
#             rz_values.append(float(columns[3]))
#             rz_err.append(float(columns[4]))
#         elif idx == 6:
#             l2_x_values.append(float(columns[3]))
#             l2_x_err.append(float(columns[4]))
#         elif idx == 7:
#             l2_y_values.append(float(columns[3]))
#             l2_y_err.append(float(columns[4]))
#         elif idx == 10:
#             l2_rz_values.append(float(columns[3]))
#             l2_rz_err.append(float(columns[4]))
#         elif idx == 11:
#             l3_x_values.append(float(columns[3]))
#             l3_x_err.append(float(columns[4]))
#         elif idx == 12:
#             l3_y_values.append(float(columns[3]))
#             l3_y_err.append(float(columns[4]))
#         elif idx == 15:
#             l3_rz_values.append(float(columns[3]))
#             l3_rz_err.append(float(columns[4]))

# for filename in filenames:
#     with open(filename, 'r') as file:
#         lines = file.readlines()
#     for idx, line in enumerate(lines):
#         columns = line.split()
#         print(columns)
#         if idx == 1:
#             x_values.append(float(columns[3]))
#             x_err.append(float(columns[4]))
#         elif idx == 2:
#             y_values.append(float(columns[3]))
#             y_err.append(float(columns[4]))
#         elif idx == 3:
#             rx_values.append(float(columns[3]))
#             rx_err.append(float(columns[4]))
#         elif idx == 4:
#             ry_values.append(float(columns[3]))
#             ry_err.append(float(columns[4]))
#         elif idx == 5:
#             rz_values.append(float(columns[3]))
#             rz_err.append(float(columns[4]))

# print("x_values:", x_values)
# print("y_values:", y_values)
# print("rx_values:", rx_values)
# print("ry_values:", ry_values)
# print("rz_values:", rz_values)
# print("x_err:", x_err)
# print("y_err:", y_err)
# print("rx_err:", rx_err)
# print("ry_err:", ry_err)
# print("rz_err:", rz_err)

# iter = list(range(1,len(filenames)+1))

# fig_11 = plt.figure(figsize=(10, 6))
# plt.errorbar(iter, x_values, yerr=x_err, fmt='o', label='layer_0_x_corrections', markersize=8)
# plt.errorbar(iter, y_values, yerr=y_err, fmt='o', label='layer_0_y_corrections', markersize=8)
# plt.axhline(y=0, color='r', linestyle='--')
# plt.ylabel('Millepde Correction (mm)', fontsize=16)
# plt.xlabel('Iteration', fontsize=16)
# plt.legend(fontsize=14)
# p.savefig(fig_11)
# plt.close(fig_11)

# fig_22 = plt.figure(figsize=(10, 6))
# # plt.errorbar(iter, rx_values, yerr=rx_err, fmt='o', label='x_rot_corrections', markersize=8)
# # plt.errorbar(iter, ry_values, yerr=ry_err, fmt='o', label='y_rot_corrections', markersize=8)
# plt.errorbar(iter, rz_values, yerr=rz_err, fmt='o', label='layer_0_z_rot_corrections', markersize=8)
# #draw red line at y = 0
# plt.axhline(y=0, color='r', linestyle='--')
# plt.ylabel('Millepde Correction (radians)', fontsize=16)
# plt.xlabel('Iteration', fontsize=16)
# plt.legend(fontsize=14)
# p.savefig(fig_22)
# plt.close(fig_22)

# fig_33 = plt.figure(figsize=(10, 6))
# plt.errorbar(iter, l2_x_values, yerr=l2_x_err, fmt='o', label='layer_1_x_corrections', markersize=8)
# plt.errorbar(iter, l2_y_values, yerr=l2_y_err, fmt='o', label='layer_1_y_corrections', markersize=8)
# plt.axhline(y=0, color='r', linestyle='--')
# plt.ylabel('Millepde Correction (mm)', fontsize=16)
# plt.xlabel('Iteration', fontsize=16)
# plt.legend(fontsize=14)
# p.savefig(fig_33)
# plt.close(fig_33)

# fig_44 = plt.figure(figsize=(10, 6))
# plt.errorbar(iter, l2_rz_values, yerr=l2_rz_err, fmt='o', label='layer_1_z_rot_corrections', markersize=8)
# plt.axhline(y=0, color='r', linestyle='--')
# plt.ylabel('Millepde Correction (radians)', fontsize=16)
# plt.xlabel('Iteration', fontsize=16)
# plt.legend(fontsize=14)
# p.savefig(fig_44)
# plt.close(fig_44)

# fig_55 = plt.figure(figsize=(10, 6))
# plt.errorbar(iter, l3_x_values, yerr=l3_x_err, fmt='o', label='layer_2_x_corrections', markersize=8)
# plt.errorbar(iter, l3_y_values, yerr=l3_y_err, fmt='o', label='layer_2_y_corrections', markersize=8)
# plt.axhline(y=0, color='r', linestyle='--')
# plt.ylabel('Millepde Correction (mm)', fontsize=16)
# plt.xlabel('Iteration', fontsize=16)
# plt.legend(fontsize=14)
# p.savefig(fig_55)
# plt.close(fig_55)

# fig_66 = plt.figure(figsize=(10, 6))
# plt.errorbar(iter, l3_rz_values, yerr=l3_rz_err, fmt='o', label='layer_2_z_rot_corrections', markersize=8)
# plt.axhline(y=0, color='r', linestyle='--')
# plt.ylabel('Millepde Correction (radians)', fontsize=16)
# plt.xlabel('Iteration', fontsize=16)
# plt.legend(fontsize=14)
# p.savefig(fig_66)
# plt.close(fig_66)