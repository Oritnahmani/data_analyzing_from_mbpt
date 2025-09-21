import numpy as np
import itertools
import scipy
import matplotlib.pyplot as plt
import scipy.constants
import h5py
from green_mbtools.pesto import mb
from mbanalysis import ir
# import sys
# import os
# current_dir = os.path.dirname('/home/orit/VS_codes/green-mbtools/')
# subfolder_path = os.path.join(current_dir, 'mbanalysis')
# sys.path.append(subfolder_path)
# import ir




def read_GW_file(inputh5_path):
    with h5py.File(inputh5_path, 'r') as f:
        it = f["iter"][()]

        mu = f['iter' + str(it) + '/mu'][()]
        G_tau = f['iter' + str(it) + '/G_tau/data'][()].view(complex)
        sigma_1 = f['iter' + str(it) + '/Sigma1'][()]
        selfenergy = f['iter' + str(it) + '/Selfenergy/data'][()].view(complex)
    return(mu , G_tau ,sigma_1, selfenergy)
    # return(mu)

def read_H_k(inputh5_path):
    with h5py.File(inputh5_path, 'r') as f:
        H_k = f['HF/H-k'][()]
    return(H_k)

def fourier_transform(selfenergy,GW_result_path,tau_grid_path):
    with h5py.File(GW_result_path, 'r') as f:
        # ir_file = tau_grid_path
        it = f["iter"][()]
        tau_mesh = f["iter" + str(it) + "/G_tau/mesh"][()]
    beta = tau_mesh[-1]
    nts = tau_mesh.shape[0]
    my_ir = ir.IR_factory(beta, tau_grid_path)
    selfenergy_iw = my_ir.tau_to_w(selfenergy)
    return(beta, selfenergy_iw)

def dyson_omega_to_green(beta, selfenergy_iw, tau_grid_path):
    my_ir = ir.IR_factory(beta, tau_grid_path)
    omegas = my_ir.wsample
    # G_w_inverse = np.empty_like(selfenergy_iw[0])
    # print(G_w_inverse)
    G_w = np.empty_like(selfenergy_iw)

    for omega in range(selfenergy_iw.shape[0]):
        for j in range(selfenergy_iw.shape[1]):
            for k in range(selfenergy_iw.shape[2]):
                print(G_w.shape)
                print(selfenergy_iw.shape)
                G_w[omega][j][k] =np.linalg.inv(-selfenergy_iw[omega][j][k])








if __name__ == '__main__':
    tau_grid_path = '/home/orit/VS_codes/Data/1e4.h5'
    GW_result_path = '/home/orit/VS_codes/green-mbtools/tests/test_data/H2_GW/sim.h5'
    inputh5_path = '/home/orit/VS_codes/green-mbtools/tests/test_data/H2_GW/input.h5'
    # mu = read_GW_file(inputh5_path)

    mu , G_tau ,sigma_1 , selfenergy = read_GW_file(GW_result_path)
    # my_ir = ir.IR_factory(beta, tau_grid_path)
    # H_k = read_H_k(inputh5_path)
    beta, selfenergy_iw = fourier_transform(selfenergy,GW_result_path,tau_grid_path)
    dyson_omega_to_green(beta, selfenergy_iw, tau_grid_path)