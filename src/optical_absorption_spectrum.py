import argparse

def parse_args() -> any:
    """
    Parses system arguments.
    Returns args: contains sys arguments added by user.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-gs", "--ground-state-poscar", type=str,
                        help="a ground state POSCAR file")
    parser.add_argument("-es", "--excited-state-poscar", type=str,
                        help="an excited state POSCAR file")
    parser.add_argument("-ph", "--phonon-file", type=str,
                        help="a ground state qpoints.yaml file")
    parser.add_argument("-temp", "--temperature", type=float,
                        help="temperature (K)")
    parser.add_argument("-zpl", "--zpl-energy", type=float,
                        help="zero-point line energy (eV)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    svmi = get_vib_modes(args.phonon_file)
    disp_arr = get_displacements(args.ground_state_poscar, args.excited_state_poscar)
    mass_arr = get_ground_state_masses(args.ground_state_poscar)
    sov = SystemOptimizedValues(svmi, mass_arr, args.zpl_energy)

    partialhrf = partial_hrf(svmi, mass_arr, sov.diag_masses, disp_arr)
    hrfspectrum = spectral_function(partialhrf, sov.ngrids, sov.energy, sov.phonE_eV)
    genfunction = generating_function(sov.total_time, sov.energy, sov.dE, hrfspectrum, args.temperature)
    plspectrum = line_shape(genfunction, sov.T)

    alg_result = ResultData()
    alg_result.optical_absorption_spectrum = plspectrum
    alg_result.photon_e = sov.photon_energy

    save_plot(args.zpl_energy, alg_result, show=True, write=True)