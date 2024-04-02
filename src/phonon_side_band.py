
import csv
import yaml
from pathlib import Path
import numpy as np
from ase import geometry
from ase.io import read

import constants as const
from system_optimized_values import SystemOptimizedValues

class SystemVibrationModesInfo:
    def __init__(self,
                 phonon_vibrational_modes: np.array,
                 phonon_energy_thz: np.array,
                 number_of_atoms: int,
                 number_of_modes: int):
        """Returns the vibrational modes related information."""
        self.phonon_vibrational_modes = phonon_vibrational_modes
        self.phonon_energy = phonon_energy_thz
        self.num_atoms = number_of_atoms
        self.num_modes = number_of_modes

def get_vib_modes(qpoint_yaml_file: Path) -> SystemVibrationModesInfo:
    """
    Parses the YAML file at the given path to extract vibration modes
    and eigenmodes information.
    
    Args:
        qpoint_yaml_file (Path): The path to a qpoints.yaml file.
    
    Returns:
        SystemVibrationModesInfo: An object containing vibration eigenmodes info.
    
    Raises:
        ValueError: If the file does not contain valid vibration modes information.
    """
    frequency = []
    eigenvector = []
    number_of_modes: int = 0
    number_of_atoms: int = 0
    with open(qpoint_yaml_file, 'r') as yamlfile:
        for line in yamlfile:
            
            if 'nqpoint:' in line:
                number_of_qpts = int(line.split()[-1])
                if number_of_qpts != 1:
                    print('*** Only Gamma-point phonons! ***')
                    break
            if 'natom:  ' in line:
                number_of_atoms = int(line.split()[-1])
                number_of_modes = 3 * number_of_atoms
            if 'frequency: ' in line:
                frequency.append(float(line.split()[-1]))
            if '- # atom ' in line:
                coord = []
                for i in range(3):
                    x = next(yamlfile).split()[2]
                    coord.append(float(x[:-1]))
                eigenvector.append(coord)
        vibrational_modes = np.reshape(eigenvector, (number_of_modes, -1))

    assert number_of_modes != 0 or number_of_atoms != 0, f"'{qpoint_yaml_file}' is not valid file!"

    system_vib_mode_info = SystemVibrationModesInfo(phonon_vibrational_modes=vibrational_modes,
                                                    phonon_energy_thz=frequency,
                                                    number_of_atoms=number_of_atoms,
                                                    number_of_modes=number_of_modes)
    return system_vib_mode_info


def get_displacements(gs_file: str, es_file: str):
    """Returns atomic displacement as a flattened numpy array."""
    a = read(gs_file)
    b = read(es_file)
    dists = geometry.get_distances(a.get_positions(),
                                   b.get_positions(),
                                   cell=a.get_cell(),
                                   pbc=True)
    disps_temp = []
    for i in range(len(a)):
        disps_temp.append(dists[0][i][i])
    disps = np.reshape(disps_temp, (-1, 1))
    return disps


def get_ground_state_masses(gs_file: str):
    """Takes ground state POSCAR file and
    returns atomic masses."""
    infile = read(gs_file)
    gsmass = infile.get_masses()
    return gsmass


def normalization(phon: np.array, mass: np.array):
    """Takes masses and phonon numpy array and
    returns normalized disp vectors!"""
    phon_tmp = np.reshape(phon, (-1, 3))
    disp_vec = []
    tot = []
    for i, vibfunc in enumerate(phon_tmp):
        disvec = vibfunc / np.sqrt(mass[i])
        tot.append(np.linalg.norm(disvec) ** 2)
        disp_vec.append(disvec)

    norm_vec = np.array(disp_vec) / np.sqrt(sum(tot))
    normalized = np.reshape(norm_vec, (1, -1))  # one column
    return normalized


def deltaq_mode(disp, phon, mass):
    """Takes displacement vectors, vibrational modes eigenmodes and atomic masses as numpy arrays
    and returns configurational coordinates."""
    mass_x_vibmodes = np.dot(phon, mass)
    conf_coord = np.dot(mass_x_vibmodes, disp)
    return conf_coord[0][0]


def omega(frequency_thz):
    """Takes frequency in THz unit and returns it to the angular frequency."""
    return 2 * np.pi * frequency_thz


def partial_hrf_mode(frequency_thz, config_coord_sqrt_amu_angstrom):
    """Takes configuration coordinates and vibrational mode's frequency
    and returns partial huang-rhys factor (HRF)."""
    angular_freq = const.TERA * omega(frequency_thz)
    conf_coord_pow_2 = np.dot(config_coord_sqrt_amu_angstrom, config_coord_sqrt_amu_angstrom) * sov.dq2_units
    return 1 / 2 / const.HBAR_TO_EVS * angular_freq * conf_coord_pow_2


def gaussian_fitting(x: float, miu: float, sigma: float):
    """Takes three float numbers as occurrence, mean, variance and
    returns Gaussian weighted value."""
    return 1 / np.sqrt(np.pi) / sigma * np.exp((x - miu) ** 2 / (-sigma ** 2))


def partial_hrf(svmi_obj: SystemVibrationModesInfo, masses, diag_mass, displacements):
    """
    Takes and object contains all phonon information and two array.
    """
    hrf = []
    conf_coordinate = []
    for mode in range(svmi_obj.num_modes):
        norm_phon_mode = normalization(np.array(svmi_obj.phonon_vibrational_modes[mode]), masses)
        conf_coord_mode = deltaq_mode(displacements, norm_phon_mode, diag_mass)  # vec
        # TODO: add one more return for printing out into a file later on.
        conf_coordinate.append(conf_coord_mode ** 2)
        s_m = partial_hrf_mode(svmi_obj.phonon_energy[mode], conf_coord_mode)
        hrf.append(s_m)
    hrf[0] = hrf[1] = hrf[2] = 0
    return hrf


def spectral_function(partialhrfs: list,
                      n_grids: int,
                      energy: np.array,
                      vib_mode_energy: np.array):
    """Calculates the spectral function.
    Takes list of partial HRF, number of grids, phonon energy range,
     phonon vibrational modes in eV units as numpy array."""
    spectral_hrf = np.zeros((n_grids))
    for i, value in enumerate(partialhrfs):
        for grid in range(n_grids):
            spectral_hrf[grid] += value * gaussian_fitting(energy[grid],
                                                           vib_mode_energy[i],
                                                           sigma=0.005)
    return spectral_hrf


def generating_function(total_time: np.array,
                        energy: np.array,
                        dE: float,
                        spectralfunction: np.array,
                        temperature: float):
    """Returns time-dependent generating function as a numpy array."""
    hrfspecfunct = np.zeros((sov.tgrids,), dtype=complex)
    for j, t in enumerate(total_time):
        for i, ene in enumerate(energy):
            tempfunc = temperature_function(ene, temperature)
            hrfspecfunct_p = spectralfunction[i] * tempfunc * np.exp(1j * ene * t * const.PICO / const.HBAR_TO_EVS) * dE
            hrfspecfunct_n = spectralfunction[i] * (1 + tempfunc) * np.exp(
                -1j * ene * t * const.PICO / const.HBAR_TO_EVS) * dE
            hrfspecfunct[j] += hrfspecfunct_n + hrfspecfunct_p
    return np.exp(hrfspecfunct - hrfspecfunct[0])


def line_shape(gen_function: np.array, timeperiod: float):
    """Calculates optical absorption spectrum.
    Takes the generating function as numpy array and time period as a float.
    Returns a numpy array."""
    optical_absorption_spectrum = 1 / 2 / np.pi * np.fft.ifft(gen_function)
    photon_omg = (2 * np.pi / timeperiod / const.PICO) * np.arange(sov.tgrids)
    photon_e = const.HBAR_TO_EVS * photon_omg
    scaled_absorption_spectrum = (((args.zpl_energy - photon_e) / const.HBAR_TO_EVS) ** 3) * abs(
        optical_absorption_spectrum)
    return scaled_absorption_spectrum


def temperature_function(energy, temperature):
    """Takes energy in eV and temperature in Kelvin and
    returns Boltzmann distribution for temperature effects."""
    return 1 / (np.exp(energy / const.KB / temperature) - 1)


def write_to_file(filename, x, y):
    """Writes information to the text file in two columns as x and y."""
    with open(filename, 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(zip(x, y))






