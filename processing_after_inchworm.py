import argparse
import time
import re
from pathlib import Path
import numpy as np
import scipy.interpolate
import h5py
from green_mbtools.pesto import mb
from mbanalysis import ir
# from inchworm_stuff.hdf5_to_txt import GW_result_path
from inchworm_stuff.redaing_txt import read_greenfunction_from_txt_spin, read_delta_tau_from_txt_spin, read_hopping_from_txt_spin

def read_mu(NiO_GW_h5):
    with h5py.File(NiO_GW_h5, 'r') as f:
        it = f["iter"][()]
        mu = f['iter' + str(it) + '/mu'][()]
    return mu



def interpolation(tau_original, delta_tau_original, tau_new, kind="linear"):
    tau_original = np.asarray(tau_original, dtype=float)
    tau_new = np.asarray(tau_new, dtype=float)
    delta_tau_original = np.asarray(delta_tau_original)

    Nt_new = len(tau_new)
    nspin = delta_tau_original.shape[1]
    norb  = delta_tau_original.shape[2]

    new_delta_tau = np.zeros((Nt_new, nspin, norb, norb), dtype=complex)

    for s in range(nspin):
        for i in range(norb):
            for j in range(norb):
                real_interp = scipy.interpolate.interp1d(
                    tau_original, delta_tau_original[:, s, i, j].real,
                    kind=kind, fill_value="extrapolate", assume_sorted=True
                )
                imag_interp = scipy.interpolate.interp1d(
                    tau_original, delta_tau_original[:, s, i, j].imag,
                    kind=kind, fill_value="extrapolate", assume_sorted=True
                )
                new_delta_tau[:, s, i, j] = real_interp(tau_new) + 1j * imag_interp(tau_new)

    return new_delta_tau







def fourier_transform( beta , ir_grid_path,new_delta_tau, green_tau ):

    my_ir = ir.IR_factory(beta, ir_grid_path)
    nspin = new_delta_tau.shape[1]
    norb  = new_delta_tau.shape[2]

    # Determine Nw from transforming one block (or use my_ir.wsample length)
    # We'll just allocate using my_ir.wsample:
    Nw = len(my_ir.wsample)

    delta_omega = np.zeros((Nw, nspin, norb, norb), dtype=complex)
    green_omega = np.zeros((Nw, nspin, norb, norb), dtype=complex)

    for s in range(nspin):
        delta_omega[:, s] = my_ir.tau_to_w(new_delta_tau[:, s])
        green_omega[:, s] = my_ir.tau_to_w(green_tau[:, s])
    return delta_omega, green_omega


def dyson_green_to_sigma_split_omega(beta, green_omega,number_of_orbitals, ir_grid_path,mu, delta_omega,hopping, avg_slice=None, use_weights=False):
    my_ir = ir.IR_factory(beta, ir_grid_path)
    eye = np.eye(number_of_orbitals, dtype=complex)

    Nw = green_omega.shape[0]
    nspin = green_omega.shape[1]

    selfenergy_iw = np.zeros((Nw, nspin, number_of_orbitals, number_of_orbitals), dtype=complex)

    for s in range(nspin):
        for omega in range(Nw):
            # Σ = -G^{-1} - t + (iω + μ)I - Δ
            selfenergy_iw[omega, s] = (
                -np.linalg.inv(green_omega[omega, s])
                - hopping[s]
                + (1j * my_ir.wsample[omega] + mu) * eye
                - delta_omega[omega, s]
            )
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


def save_sigma_split_to_hdf5(sigma_file, sigma_static, sigma_dynamic_iw, sigma_iw=None):
    with h5py.File(sigma_file, "w") as f:
        f.create_dataset("Sigma_static", data=sigma_static)                 # (Norb, Norb)
        f.create_dataset("Sigma_dynamic_iw", data=sigma_dynamic_iw)         # (Nw, Norb, Norb)
        if sigma_iw is not None:
            f.create_dataset("Sigma_iw", data=sigma_iw)                     # optional



def read_beta_from_seet_sbatch(sbatch_path: str | Path) -> float:
    text = Path(sbatch_path).read_text()

    # Matches:
    #   BETA=100
    #   export BETA=100
    #   BETA = 100.0
    m = re.search(r'(?m)^\s*(?:export\s+)?BETA\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*$', text)
    if not m:
        raise ValueError(f"Could not find a line like 'BETA=...' in {sbatch_path}")
    return float(m.group(1))


def build_argparser():
    ap = argparse.ArgumentParser(
        description="Wait for inchworm G files, then compute selfenergy_iw."
    )

    ap.add_argument("--sbatch-dir", type=Path, default=None,
                    help="Directory containing the SEET sbatch file (default: run-dir)")
    ap.add_argument("--sbatch-name", type=Path, default=Path("sbatch_seet"),
                    help="Filename of the SEET sbatch script inside sbatch-dir")

    ap.add_argument("--run-dir", type=Path, default=".",
                    help="Directory where inchworm output files are located.")
    
    # New: impurity control
    ap.add_argument("--nimp", type=int, required=True,
                    help="Number of impurities to process, e.g. 2")

    ap.add_argument("--impurity-pattern", type=str, default="imp_{imp}",
                    help='Pattern for impurity subdirectories under run-dir, e.g. "imp_{imp}" '
                         'gives run_dir/imp_0, run_dir/imp_1, ...')

    ap.add_argument("--time_intervals", type=Path, default="time_intervals.txt",
                    help="Path (relative to run-dir) for time_intervals.txt")
    ap.add_argument("--delta-file", type=Path, default="delta.txt",
                    help="Path (relative to run-dir) for delta.txt")
    ap.add_argument("--hopping-file", type=Path, default="hopping.txt",
                    help="Path (relative to run-dir) for hopping.txt")

    ap.add_argument("--nio-gw-h5", type=Path, default="NiO_GW_iter14.h5",
                    help="Path to NiO_GW_iter*.h5 (default: NiO_GW_iter14.h5)")
    ap.add_argument("--input-h5", type=Path, default="input.h5",
                    help="Path to input.h5 (grid info)")
    ap.add_argument("--ir-grid", type=Path, default="1e5.h5",
                    help="Path to IR grid h5 file, e.g. 1e5.h5")

    ap.add_argument("--g-dir", type=Path, default=None,
                    help="Directory containing G_{i}_{j}.dat files (default: run-dir)")
    ap.add_argument("--g-pattern", type=str, default="G_{i}_{j}.dat",
                    help='Green filename pattern, e.g. "G_{i}_{j}.dat"')

    ap.add_argument("--out-h5", type=Path, default="selfenergy_split_all.h5",
                help="Output HDF5 file (saved in run-dir).")
    
    ap.add_argument("--beta", type=float, default=None,
                help="Inverse temperature beta.")
    return ap


def run_processing(args):
    run_dir = args.run_dir.expanduser().resolve()

    def resolve_under(base: Path, p: Path) -> Path:
        p = Path(p).expanduser()
        return (base / p).resolve() if not p.is_absolute() else p.resolve()

    nio_gw_h5 = resolve_under(run_dir, args.nio_gw_h5)
    ir_grid = resolve_under(run_dir, args.ir_grid)

    sbatch_dir = resolve_under(run_dir, args.sbatch_dir) if args.sbatch_dir is not None else run_dir
    sbatch_path = (sbatch_dir / args.sbatch_name).expanduser().resolve()

    if args.beta is not None:
        beta = args.beta
    else:
        beta = read_beta_from_seet_sbatch(sbatch_path)

    mu = read_mu(str(nio_gw_h5))

    sigma_imp_list = []
    sigma_inf_list = []
    sigma_iw_list = []

    shape_dyn_ref = None
    shape_stat_ref = None
    shape_full_ref = None

    for imp in range(args.nimp):
        imp_dir = run_dir / args.impurity_pattern.format(imp=imp)

        time_filename = resolve_under(imp_dir, args.time_intervals)
        delta_file = resolve_under(imp_dir, args.delta_file)
        hopping_file = resolve_under(imp_dir, args.hopping_file)
        g_dir = resolve_under(imp_dir, args.g_dir) if args.g_dir is not None else imp_dir

        print(f"Processing impurity {imp} in {imp_dir}")

        hopping = read_hopping_from_txt_spin(str(hopping_file))   # (ns, norb, norb)
        num_orbitals = hopping.shape[1]

        green_tau, t_arr = read_greenfunction_from_txt_spin(
            time_filename=str(time_filename),
            green_path=str(g_dir),
        )

        delta_tau, tau_delta_original = read_delta_tau_from_txt_spin(
            str(delta_file),
            beta=beta,
        )

        new_delta_tau = interpolation(
            tau_delta_original,
            delta_tau,
            t_arr,
            kind="linear",
        )

        delta_omega, green_omega, my_ir = fourier_transform(
            beta,
            str(ir_grid),
            new_delta_tau,
            green_tau,
        )

        selfenergy_iw, sigma_static, sigma_dynamic_iw, _ = dyson_green_to_sigma_split_omega(
            beta=beta,
            green_omega=green_omega,
            number_of_orbitals=num_orbitals,
            ir_grid_path=str(ir_grid),
            mu=mu,
            delta_omega=delta_omega,
            hopping=hopping,
            avg_slice=None,
            use_weights=False,
        )

        if shape_dyn_ref is None:
            shape_dyn_ref = sigma_dynamic_iw.shape
            shape_stat_ref = sigma_static.shape
            shape_full_ref = selfenergy_iw.shape
        else:
            if sigma_dynamic_iw.shape != shape_dyn_ref:
                raise ValueError(
                    f"Impurity {imp} dynamic sigma has shape {sigma_dynamic_iw.shape}, "
                    f"expected {shape_dyn_ref}. Cannot stack impurities into one tensor."
                )
            if sigma_static.shape != shape_stat_ref:
                raise ValueError(
                    f"Impurity {imp} static sigma has shape {sigma_static.shape}, "
                    f"expected {shape_stat_ref}. Cannot stack impurities into one tensor."
                )
            if selfenergy_iw.shape != shape_full_ref:
                raise ValueError(
                    f"Impurity {imp} full sigma has shape {selfenergy_iw.shape}, "
                    f"expected {shape_full_ref}. Cannot stack impurities into one tensor."
                )

        sigma_iw_list.append(selfenergy_iw)       # (nomega, ns, norb, norb)
        sigma_imp_list.append(sigma_dynamic_iw)   # (nomega, ns, norb, norb)
        sigma_inf_list.append(sigma_static)       # (ns, nsorb, nsorb)

    sigma_iw_all = np.stack(sigma_iw_list, axis=0)
    sigma_imp_all = np.stack(sigma_imp_list, axis=0)
    sigma_inf_all = np.stack(sigma_inf_list, axis=0)

    out_h5 = resolve_under(run_dir, args.out_h5)
    with h5py.File(out_h5, "w") as f:
        f.create_dataset("Sigma_iw_all", data=sigma_iw_all)
        f.create_dataset("Sigma_dynamic_iw_all", data=sigma_imp_all)
        f.create_dataset("Sigma_static_all", data=sigma_inf_all)

    return sigma_imp_all, sigma_inf_all




def main():

    ap = build_argparser()
    args = ap.parse_args()
    _ = run_processing(args)
    # ap = argparse.ArgumentParser(description="Wait for inchworm G files, then compute selfenergy_iw.")
    # ap.add_argument("--sbatch-dir", type=Path, default=None,
    #             help="Directory containing the SEET sbatch file (default: run-dir)")
    # ap.add_argument("--sbatch-name", type=Path, default=Path("sbatch_seet"),
    #             help="Filename of the SEET sbatch script inside sbatch-dir")

    # ap.add_argument("--run-dir",type=Path, default=".", help="Directory where inchworm output files are located.")
    # ap.add_argument("--time_intervals",type=Path, default="time_intervals.txt", help="Path (relative to run-dir) for time_intervals.txt")
    # ap.add_argument("--delta-file",type=Path, default="delta.txt", help="Path (relative to run-dir) for delta.txt")
    # ap.add_argument("--hopping-file", type=Path, default="hopping.txt", help="Path (relative to run-dir) for hopping.txt")
    # # Mu inputs (these are in your original script; make them arguments so it works on cluster)
    # #TODO
    # ap.add_argument("--nio-gw-h5", type=Path, default="NiO_GW_iter14.h5", help="Path to NiO_GW_iter*.h5 (default: NiO_GW_iter14.h5)")
    # ap.add_argument("--input-h5", type=Path, default="input.h5", help="Path to input.h5 (grid info)")
    # # TODO
    # ap.add_argument("--ir-grid", type=Path, default="1e5.h5" , help="Path to IR grid h5 file, e.g. 1e5.h5")

    # # Green naming
    # ap.add_argument("--g-dir", type=Path, default=None,
    #             help="Directory containing G_{i}_{j}.dat files (default: run-dir)")
    # ap.add_argument("--g-pattern", type=str, default="G_{i}_{j}.dat",
    #             help='Green filename pattern, e.g. "G_{i}_{j}.dat"')


    # # Output
    # ap.add_argument("--out-npy", default="selfenergy_iw.npy", help="Output numpy file (saved in run-dir).")
    # args = ap.parse_args()


    # run_dir = args.run_dir.expanduser().resolve() 
    # def resolve(p: Path) -> Path: 
    #     p = p.expanduser() 
    #     return (run_dir / p).resolve() if not p.is_absolute() else p.resolve() 



    # time_filename = resolve(args.time_intervals) 
    # delta_file = resolve(args.delta_file) 
    # hopping_file = resolve(args.hopping_file) 
    # nio_gw_h5 = resolve(args.nio_gw_h5) 
    # input_h5 = resolve(args.input_h5) 
    # ir_grid = resolve(args.ir_grid)

    # g_dir = resolve(args.g_dir) if args.g_dir is not None else run_dir
    # print(g_dir)
    # sbatch_dir = resolve(args.sbatch_dir) if args.sbatch_dir is not None else run_dir
    # sbatch_path = (sbatch_dir / args.sbatch_name).expanduser().resolve()


 

    # beta = read_beta_from_seet_sbatch(sbatch_path)
    # mu = read_mu(str(nio_gw_h5))


    # hopping = read_hopping_from_txt_spin(str(hopping_file))   # (spin, Norb, Norb)
    # num_orbitals = hopping.shape[1]                           # Norb (per spin)

    # green_tau, t_arr = read_greenfunction_from_txt_spin(
    #     time_filename=str(time_filename),
    #     green_path=str(g_dir),
    # )  # (Nt, spin, Norb, Norb), (Nt,)

    # delta_tau, tau_delta_original = read_delta_tau_from_txt_spin(
    #     str(delta_file),
    #     beta=beta
    # )  # (Nt_delta, spin, Norb, Norb), (Nt_delta,)

    # new_delta_tau =  interpolation(tau_delta_original, delta_tau, t_arr, kind="linear")


    # delta_omega, green_omega = fourier_transform(beta, str(ir_grid), new_delta_tau, green_tau)

    # selfenergy_iw, sigma_static, sigma_dynamic_iw, my_ir = dyson_green_to_sigma_split_omega(
    # beta=beta,
    # green_omega=green_omega,
    # number_of_orbitals=num_orbitals,
    # ir_grid_path=str(ir_grid),
    # mu=mu,
    # delta_omega=delta_omega,
    # hopping=hopping,
    # avg_slice=None,          # or e.g. slice(10, 200)
    # use_weights=False)


    # sigma_file = run_dir / "selfenergy_split.h5"
    # save_sigma_split_to_hdf5(
    #     sigma_file,
    #     sigma_static,
    #     sigma_dynamic_iw,
    #     sigma_iw=selfenergy_iw
    # )



 


        



if __name__ == "__main__":
    main()





