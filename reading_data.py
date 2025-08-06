import numpy as np
import itertools
import scipy
import matplotlib.pyplot as plt
import scipy.constants
import h5py

def read_that_file(file_path):
    with h5py.File(file_path, 'r') as f:
        # group = f['G_tau_two_particles']
        # data = 
        mu = f['iter1/mu'][()]
        G_tau = f['iter1/G_tau/data'][()]
        sigma_1 = f['iter1/Sigma1'][()]
        selenergy = f['iter1/Selfenergy/data'][()]
    return(mu , G_tau ,sigma_1, selenergy)



if __name__ == '__main__':
    mu , G_tau ,sigma_1 , selenergy = read_that_file('/home/orit/VS_codes/Analysis_GW/green-mbtools/examples/NiO_GW.h5')
    print(selenergy)