import numpy as np
import itertools
import scipy
import matplotlib.pyplot as plt
import scipy.constants
import h5py
from green_mbtools.pesto import mb
from mbanalysis import ir

def read_delta_tau_from_txt(delta_file,t_arr,number_of_orbitals):
    delta_tau = np.zeros((t_arr.shape[0],number_of_orbitals, number_of_orbitals), dtype=complex)
    with open(delta_file) as k:
        for raw_line in k:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            l_str, i_str, m_str, ij_str_re, ij_str_im = line.split()
            l = int(l_str)
            i = int(i_str)
            m = int(m_str)
            delta_tau[l, int(i), int(m)] = complex(float(ij_str_re) + 1j * float(ij_str_im))
    return delta_tau

def interpolation(tau_original, delta_tau_original, tau_new):
    beta = tau_original[-1]
    

def fourier_transform(delta_tau, tau, beta):
    my_ir = ir.IR_factory(beta, None)
    delta_ir_coeffs = my_ir.tau_to_ir(delta_tau, tau)
    delta_omega = my_ir.ir_to_w(delta_ir_coeffs)
    return delta_omega


if __name__ == '__main__':
    # number_of_orbitals = 2
    time_filename = '/home/orit/VS_codes1/example/time_intervals.txt'
    delta_file = '/home/orit/VS_codes1/example/delta.txt'
    delta_tau = read_delta_tau_from_txt(delta_file, t_arr, number_of_orbitals)
