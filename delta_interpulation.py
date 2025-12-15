import numpy as np
import itertools
import scipy
import matplotlib.pyplot as plt
import scipy.constants
import h5py
from green_mbtools.pesto import mb
from mbanalysis import ir

def interpolation(tau_original, delta_tau_original, tau_new):
    beta = tau_original[-1]
    my_ir = ir.IR_factory(beta, None)
    G_ir_coeffs = my_ir.tau_to_ir(G_tau_original, tau_original)
    G_tau_new = my_ir.ir_to_tau(G_ir_coeffs, tau_new)
    return G_tau_new