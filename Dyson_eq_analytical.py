import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.pop("SLURM_JOB_CPUS_PER_NODE", None)
import numpy as np
import itertools
import scipy
import matplotlib.pyplot as plt
import scipy.constants
import h5py
from green_mbtools.pesto import mb
from mbanalysis import ir



def read_GW_file(sim_h5):
    with h5py.File(inputh5_path, 'r') as f:
        ir_list = f["/grid/ir_list"][()]
        weight = f["/grid/weight"][()]
        index = f["/grid/index"][()]
        conj_list = f["grid/conj_list"][()]
    with h5py.File(sim_h5, 'r') as f:
        it = f["iter"][()]
        mur = f['iter' + str(it) + '/mu'][()]
        G_taur = f['iter' + str(it) + '/G_tau/data'][()].view(complex)
        sigma_1r = f['iter' + str(it) + '/Sigma1'][()]
        selfenergyr = f['iter' + str(it) + '/Selfenergy/data'][()].view(complex)
        tau = f['iter' + str(it) + '/G_tau/mesh'][()]
        mu = mur
        G_tau = mb.to_full_bz(G_taur, conj_list, ir_list, index, 2)
        sigma_1 = mb.to_full_bz(sigma_1r, conj_list, ir_list, index, 1)
        selfenergy = mb.to_full_bz(selfenergyr, conj_list, ir_list, index, 2)
    return(mu , G_tau ,sigma_1, selfenergy,tau )
    # return(mu)

def read_H_k(inputh5_path):
    with h5py.File(inputh5_path, 'r') as f:
        H_k = f['HF/H-k'][()].view(complex)
        S_k = f['HF/S-k'][()].view(complex)
    if H_k.shape[-1] == 1:
        H_k = H_k[..., 0]
    if S_k.shape[-1] == 1:
        S_k = S_k[..., 0]
    return(H_k, S_k)

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

def dyson_omega_to_green(beta, selfenergy_iw, sigma_1, tau_grid_path,H_k,S_k,mu):
    my_ir = ir.IR_factory(beta, tau_grid_path)
    # G_w_inverse = np.empty_like(selfenergy_iw[0])
    # print(G_w_inverse)
    G_w = np.empty_like(selfenergy_iw)
    for omega in range(selfenergy_iw.shape[0]):
        for l in range(selfenergy_iw.shape[1]):
            for k in range(selfenergy_iw.shape[2]):
                G_w[omega,l,k,:,:] =np.linalg.inv(-selfenergy_iw[omega,l,k,:,:] -  sigma_1[l,k,:,:] - H_k[l,k,:,:] + (1j * my_ir.wsample[omega] + mu) * S_k[l,k,:,:])
        # G_w[omega,0,0,:,:] =np.linalg.inv(-selfenergy_iw[omega,0,0,:,:] + (H_k[0,0,:,:] + S_k[0,0,:,:]))

    G_tau_dyson = my_ir.w_to_tau(G_w)
    return(G_tau_dyson)


def dyson_green_to_sigma_with_delta(beta, selfenergy_iw, sigma_1, tau_grid_path,H_k,S_k,mu, delta_omega):
    my_ir = ir.IR_factory(beta, tau_grid_path)
    G_w = np.empty_like(selfenergy_iw)
    for omega in range(selfenergy_iw.shape[0]):
        for l in range(selfenergy_iw.shape[1]):
            for k in range(selfenergy_iw.shape[2]):
                selfenergy_iw[omega,l,k,:,:] = - np.linalg.inv(G_w[omega,l,k,:,:]) - H_k[l,k,:,:] + (1j * my_ir.wsample[omega] + mu) * S_k[l,k,:,:] - delta_omega[omega,l,k,:,:]
                # G_w[omega,l,k,:,:] =np.linalg.inv(-selfenergy_iw[omega,l,k,:,:] -  sigma_1[l,k,:,:] - H_k[l,k,:,:] + (1j * my_ir.wsample[omega] + mu) * S_k[l,k,:,:] + delta_omega[omega,l,k,:,:])
    G_tau_dyson = my_ir.w_to_tau(G_w)
    return(G_tau_dyson)




if __name__ == '__main__':
    tau_grid_path = '/home/orit/VS_codes1/green-mbtools/tests/test_data/ir_grid/1e4.h5'
    GW_result_path = '/home/orit/VS_codes1/green-mbtools/tests/test_data/H2_GW/sim.h5'
    inputh5_path = '/home/orit/VS_codes1/green-mbtools/tests/test_data/H2_GW/input.h5'
    # mu = read_GW_file(inputh5_path)

    mu , G_tau ,sigma_1 , selfenergy,tau = read_GW_file(GW_result_path)

    H_k,S_k = read_H_k(inputh5_path)
    beta, selfenergy_iw = fourier_transform(selfenergy,GW_result_path,tau_grid_path)
    G_tau_dyson = dyson_omega_to_green(beta, selfenergy_iw, sigma_1, tau_grid_path, H_k, S_k, mu)


    G_diff = np.zeros(tau.shape[0])
    for i in range(tau.shape[0]):
        G_diff[i] = np.max(np.abs(G_tau[i, 0, 0, :, :] - G_tau_dyson[i, 0, 0, :, :]))
    plt.plot (tau, G_diff) 
    plt.savefig('G_diff.png') 
    plt.clf()  
    plt.plot(tau, G_tau[:,0,0,0,0].real,  label='G_tau from GW')
    plt.plot(tau, G_tau_dyson[:,0,0,0,0].real, label='G_tau from Dyson')
    plt.legend()
    plt.savefig('G_tau_comparison.png')
 
