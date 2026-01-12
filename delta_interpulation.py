import argparse
import time
import re
from pathlib import Path
import numpy as np
import scipy.interpolate
import h5py
from green_mbtools.pesto import mb
from mbanalysis import ir
from inchworm_stuff.redaing_txt import read_greenfunction_from_txt, read_delta_tau_from_txt, read_hopping_from_txt

def read_mu(NiO_GW_h5,inputh5_path):
    with h5py.File(inputh5_path, 'r') as f:
        ir_list = f["/grid/ir_list"][()]
        weight = f["/grid/weight"][()]
        index = f["/grid/index"][()]
        conj_list = f["grid/conj_list"][()]
    with h5py.File(NiO_GW_h5, 'r') as f:
        it = f["iter"][()]
        mu = f['iter' + str(it) + '/mu'][()]
        sigma_1r = f['iter' + str(it) + '/Sigma1'][()]
        sigma_1 = mb.to_full_bz(sigma_1r, conj_list, ir_list, index, 1)
    return mu, sigma_1

def interpolation(tau_original, delta_tau_original, tau_new):
    # beta = tau_original[-1]
    new_delta_tau = np.zeros((len(tau_new), delta_tau_original.shape[1], delta_tau_original.shape[2]), dtype=complex)
    for i in range(delta_tau_original.shape[1]):
        for j in range(delta_tau_original.shape[2]):
            real_interp = scipy.interpolate.interp1d(tau_original, delta_tau_original[:, i, j].real, kind='cubic', fill_value="extrapolate")
            imag_interp = scipy.interpolate.interp1d(tau_original, delta_tau_original[:, i, j].imag, kind='cubic', fill_value="extrapolate")
            new_delta_tau[:, i, j] = real_interp(tau_new) + 1j * imag_interp(tau_new)
    return new_delta_tau



# def read_g_tau_from_txt(g_tau_file):
#     g_tau = []
#     with open(g_tau_file) as f:
#         for line in f:
#             if line.startswith("#"):
#                 continue
#             g_tau.append(np.array([float(x) for x in line.split()]))
#     return np.array(g_tau)


def fourier_transform(new_delta_tau, beta,green_tau, ir_grid_path):
    my_ir = ir.IR_factory(beta, ir_grid_path)
    delta_omega = my_ir.tau_to_w(new_delta_tau)
    green_omega = my_ir.tau_to_w(green_tau)
    return delta_omega, green_omega


def dyson_green_to_sigma_with_delta(beta, green_omega,number_of_orbitals, ir_grid_path,mu, delta_omega,hopping, avg_slice=None, use_weights=False):
    selfenergy_iw = np.zeros((green_omega.shape[0], number_of_orbitals, number_of_orbitals), dtype=complex)
    my_ir = ir.IR_factory(beta, ir_grid_path)
    eye = np.eye(number_of_orbitals, dtype=complex)
    for omega in range(green_omega.shape[0]):
        selfenergy_iw[omega,:,:] = - np.linalg.inv(green_omega[omega,:,:]) - hopping + (1j * my_ir.wsample[omega] + mu) * eye - delta_omega[omega,:,:]
    if avg_slice is None:
        sigma_block = selfenergy_iw
        wsample_block = my_ir.wsample
    else:
        sigma_block = selfenergy_iw[avg_slice, :, :]
        wsample_block = my_ir.wsample[avg_slice]
    if use_weights:
        # A common choice is to downweight large |w| or emphasize them; pick something sensible.
        # Here: weights ~ 1/|w| (avoid div by 0 though fermionic w never 0).
        wts = 1.0 / np.abs(wsample_block)
        wts = wts / np.sum(wts)
        sigma_static = np.tensordot(wts, sigma_block, axes=(0, 0))  # (Norb, Norb)
    else:
        sigma_static = np.mean(sigma_block, axis=0)

    sigma_dynamic_iw = selfenergy_iw - sigma_static[None, :, :]
    return selfenergy_iw, sigma_static, sigma_dynamic_iw, my_ir





def snapshot(path: Path):
    st = path.stat()
    return (st.st_size, st.st_mtime_ns)

def wait_for_files(files, poll_s=5.0, stable_checks=2, stable_interval_s=2.0):
    """
    Wait until all files exist and don't change across stable_checks snapshots.
    Also require non-empty.
    """
    files = [Path(f) for f in files]
    while True:
        if all(f.exists() and f.stat().st_size > 0 for f in files):
            # stability check
            ok = True
            last = [snapshot(f) for f in files]
            for _ in range(stable_checks):
                time.sleep(stable_interval_s)
                cur = [snapshot(f) for f in files]
                if cur != last:
                    ok = False
                    break
                last = cur
            if ok:
                return
        time.sleep(poll_s)

def find_g_files(run_dir: Path, orbitals: int, g_glob: str):
    """
    Returns list of expected Green's function files to wait for.
    Example g_glob: 'G_{i}_{j}.txt' or 'G_{i}_{j}'.
    """
    expected = []
    for i in range(orbitals):
        for j in range(orbitals):
            name = g_glob.format(i=i, j=j)
            expected.append(run_dir / name)
    return expected


def save_sigma_split_to_hdf5(h5_path, sigma_static, sigma_dynamic_iw, sigma_iw=None):
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("Sigma_static", data=sigma_static)                 # (Norb, Norb)
        f.create_dataset("Sigma_dynamic_iw", data=sigma_dynamic_iw)         # (Nw, Norb, Norb)
        if sigma_iw is not None:
            f.create_dataset("Sigma_iw", data=sigma_iw)                     # optional



def main():
    ap = argparse.ArgumentParser(description="Wait for inchworm G files, then compute selfenergy_iw.")
    ap.add_argument("--run-dir", default=".", help="Directory where inchworm output files are located.")
    ap.add_argument("--beta", type=float, required=True)
    ap.add_argument("--orbitals", type=int, required=True)
    ap.add_argument("--time-intervals", default="time_intervals.txt", help="Path (relative to run-dir) for time_intervals.txt")
    ap.add_argument("--delta-file", default="delta.txt", help="Path (relative to run-dir) for delta.txt")
    ap.add_argument("--hopping-file", default="hopping.txt", help="Path (relative to run-dir) for hopping.txt")

    # Mu inputs (these are in your original script; make them arguments so it works on cluster)
    ap.add_argument("--nio-gw-h5", required=True, help="Path to NiO_GW_iter*.h5")
    ap.add_argument("--input-h5", required=True, help="Path to input.h5 (grid info)")
    ap.add_argument("--ir-grid", required=True, help="Path to IR grid h5 file, e.g. 1e5.h5")

    # Green naming
    ap.add_argument("--g-pattern", default="G_{i}_{j}.txt",
                    help='Expected file name pattern for Green outputs. Use {i} and {j}, e.g. "G_{i}_{j}.txt"')

    # Output
    ap.add_argument("--out-npy", default="selfenergy_iw.npy", help="Output numpy file (saved in run-dir).")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()

    # 1) Wait until ALL G_{i}_{j} files exist & stable
    g_files = find_g_files(run_dir, args.orbitals, args.g_pattern)
    print(f"[watch] Waiting for {len(g_files)} Green files like: {run_dir / args.g_pattern.format(i=0,j=0)}")
    wait_for_files(g_files, poll_s=5.0, stable_checks=2, stable_interval_s=2.0)
    print("[watch] Green files present and stable.")

    # 2) Compute selfenergy
    mu, sigma_1 = read_mu(args.nio_gw_h5, args.input_h5)

    time_filename = run_dir / args.time_intervals
    delta_file = run_dir / args.delta_file
    hopping_file = run_dir / args.hopping_file

    green_tau, t_arr = read_greenfunction_from_txt(args.orbitals, str(time_filename), str(run_dir))
    delta_tau = read_delta_tau_from_txt(str(delta_file), t_arr, args.orbitals)

    new_delta_tau = interpolation(t_arr, delta_tau, t_arr)
    delta_omega, green_omega, my_ir = fourier_transform(args.beta, args.ir_grid, new_delta_tau, green_tau)

    hopping = read_hopping_from_txt(str(hopping_file), args.orbitals)
    selfenergy_iw = dyson_green_to_sigma_with_delta(
        args.beta, green_omega, args.orbitals, mu, delta_omega, hopping, my_ir
    )

    out_path = run_dir / args.out_npy
    np.save(out_path, selfenergy_iw)
    print(f"[save] {out_path}  shape={selfenergy_iw.shape} dtype={selfenergy_iw.dtype}")

if __name__ == "__main__":
    main()





# if __name__ == '__main__':
#     beta = 100.0
#     number_of_orbitals = 4
#     NiO_GW_h5 = '/home/orit/VS_codes1/NiO_GW_iter14.h5'
#     inputh5_path = '/home/orit/VS_codes1/input.h5'
#     time_filename = '/home/orit/VS_codes1/data_analyzing_from_mbpt/time_intervals.txt'
#     ir_grid_path = '/home/orit/VS_codes1/data_analyzing_from_mbpt/1e5.h5'
#     delta_file = '/home/orit/VS_codes1/example/delta.txt'
#     hopping_file = '/home/orit/VS_codes1/example/hopping.txt'
#     mu, sigma_1 = read_mu(NiO_GW_h5,inputh5_path)
#     green_tau, t_arr = read_greenfunction_from_txt(number_of_orbitals, time_filename,'/home/orit/VS_codes1/example')
#     delta_tau = read_delta_tau_from_txt(delta_file, t_arr, number_of_orbitals)
#     new_delta_tau = interpolation(t_arr, delta_tau, t_arr)
#     delta_omega, green_omega = fourier_transform(new_delta_tau, beta, green_tau)
#     hopping = read_hopping_from_txt(hopping_file, number_of_orbitals)
#     selfenergy_iw = dyson_green_to_sigma_with_delta(beta, green_omega, number_of_orbitals, ir_grid_path, mu, delta_omega, hopping)
