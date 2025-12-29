import numpy as np
import itertools
import scipy
import matplotlib.pyplot as plt
import scipy.constants
import h5py
from green_mbtools.pesto import mb
from mbanalysis import ir
from data_analyzing_from_mbpt.Dyson_eq_analytical import dyson_omega_to_green_with_delta, read_GW_file, read_H_k
from inchworm_stuff.redaing_txt import read_greenfunction_from_txt, read_delta_tau_from_txt

def read_mu(NiO_GW_h5):
    with h5py.File(NiO_GW_h5, 'r') as f:
        it = f["iter"][()]
        mu = f['iter' + str(it) + '/mu'][()]
    return mu

def interpolation(tau_original, delta_tau_original, tau_new):
    # beta = tau_original[-1]
    new_delta_tau = np.zeros((len(tau_new), delta_tau_original.shape[1], delta_tau_original.shape[2]), dtype=complex)
    for i in range(delta_tau_original.shape[1]):
        for j in range(delta_tau_original.shape[2]):
            real_interp = scipy.interpolate.interp1d(tau_original, delta_tau_original[:, i, j].real, kind='cubic', fill_value="extrapolate")
            imag_interp = scipy.interpolate.interp1d(tau_original, delta_tau_original[:, i, j].imag, kind='cubic', fill_value="extrapolate")
            new_delta_tau[:, i, j] = real_interp(tau_new) + 1j * imag_interp(tau_new)
    return new_delta_tau


def read_g_tau_from_txt(g_tau_file):
    g_tau = []
    with open(g_tau_file) as f:
        for line in f:
            if line.startswith("#"):
                continue
            g_tau.append(np.array([float(x) for x in line.split()]))
    return np.array(g_tau)


def fourier_transform(new_delta_tau, t_arr, beta,green_tau):
    my_ir = ir.IR_factory(beta, ir_grid_path)
    print("ir tau_mesh shape:", my_ir.tau_mesh.shape)
    delta_omega = my_ir.tau_to_w(new_delta_tau)
    green_omega = my_ir.tau_to_w(green_tau)
    return delta_omega, green_omega


if __name__ == '__main__':
    beta = 100.0
    number_of_orbitals = 4
    NiO_GW_h5 = '/home/orit/VS_codes1/data_analyzing_from_mbpt/NiO_GW_iter14.h5'
    inputh5_path = '/home/orit/VS_codes1/data_analyzing_from_mbpt/input.h5'
    time_filename = '/home/orit/VS_codes1/data_analyzing_from_mbpt/time_intervals.txt'
    ir_grid_path = '/home/orit/VS_codes1/data_analyzing_from_mbpt/1e5.h5'
    delta_file = '/home/orit/VS_codes1/example/delta.txt'
    green_tau, t_arr = read_greenfunction_from_txt(number_of_orbitals, time_filename)
    print("t_arr shape:", t_arr.shape)
    delta_tau = read_delta_tau_from_txt(delta_file, t_arr, number_of_orbitals)
    new_delta_tau = interpolation(t_arr, delta_tau, t_arr)
    print(new_delta_tau.shape)
    delta_omega, green_omega = fourier_transform(delta_tau, t_arr, beta, green_tau)
    H_k,S_k = read_H_k(inputh5_path)