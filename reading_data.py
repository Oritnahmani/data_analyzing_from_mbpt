import numpy as np
import itertools
import scipy
import matplotlib.pyplot as plt
import scipy.constants
import h5py

def read_GW_file(NiGWh5_path):
    with h5py.File(NiGWh5_path, 'r') as f:
        mu = f['iter1/mu'][()]
        G_tau = f['iter1/G_tau/data'][()]
        sigma_1 = f['iter1/Sigma1'][()]
        selenergy = f['iter1/Selfenergy/data'][()]
    return(mu , G_tau ,sigma_1, selenergy)

def read_H_k(inputh5_path):
    with h5py.File(inputh5_path, 'r') as f:
        H_k = f['HF/H-k'][()]
    return(H_k)


if __name__ == '__main__':
    inputh5_path = '/home/orit//green_fun/data_analyzing_from_mbpt/input.h5'
    mu , G_tau ,sigma_1 , selenergy = read_GW_file('/home/orit/VS_codes/Analysis_GW/green-mbtools/examples/NiO_GW.h5')
    H_k = read_H_k(inputh5_path)
