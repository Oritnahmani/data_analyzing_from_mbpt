import numpy as np
import itertools
import scipy
import matplotlib.pyplot as plt
import scipy.constants
import h5py
from mbanalysis import ir


def read_GW_file(NiGWh5_path):
    with h5py.File(NiGWh5_path, 'r') as f:
        mu = f['iter1/mu'][()]
        G_tau = f['iter' + str(it) + '/G_tau/data'][()].view(complex)
        sigma_1 = f['iter' + str(it) + '/Sigma1'][()]
        selfenergy = f['iter' + str(it) + '/Selfenergy/data'][()]
    return(mu , G_tau ,sigma_1, selfenergy)

def read_H_k(inputh5_path):
    with h5py.File(inputh5_path, 'r') as f:
        H_k = f['HF/H-k'][()]
    return(H_k)

def fourier_transform(selfenergy,ir_f,inputh5_path,tau_grid_path):
    with h5py.File(inputh5_path, 'r') as f:
        ir_file = tau_grid_path
        it = f["iter"][()]
        tau_mesh = f["iter" + str(it) + "/G_tau/mesh"][()]
    beta = tau_mesh[-1]
    nts = tau_mesh.shape[0]
    my_ir = ir.IR_factory(beta, ir_file)
    selfenergy_iw = my_ir.tau_to_w(selfenergy)
    return(G_iw)

# def 






if __name__ == '__main__':
    tau_grid_path = '/home/orit/VS_codes/Data/1e5.h5'
    beta = tau_mesh[-1]
    inputh5_path = '/home/orit/VS_codes/Data/1e5.h5'
    mu , G_tau ,sigma_1 , selefnergy = read_GW_file(inputh5_path)
    H_k = read_H_k(inputh5_path)
    G_iw = fourier_transform(selfenergy,ir_f,inputh5_path,tau_grid_path)
