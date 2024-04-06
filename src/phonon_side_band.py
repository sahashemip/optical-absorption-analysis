
from pathlib import Path
import numpy as np
from ase import geometry
from ase.io import read

import constants as const
import utils

from system_optimized_values import SystemOptimizedValues as sov
from parse_valid_yaml import SystemVibrationModesInfo

class PhononSideBand:
    def __init__(self,
                 ground_state_file: Path,
                 excited_state_file: Path,
                 system_optimized_values_obj: SystemVibrationModesInfo,
                 accepted_cell_deviation: float = 0.1
                 ) -> None:
        """
        Initialize PhononSideBand with paths to the ground and excited state POSCAR
        files and SystemVibrationModesInfo object.

        Args:
        - ground_state_file (Path): Path to the ground state POSCAR file.
        - excited_state_file (Path): Path to the excited state POSCAR file.
        - system_optimized_values_obj (SystemVibrationModesInfo): Stores vibration
            modes information including eigenvectors, eigenvalues, and number of atoms.
        - accepted_cell_deviation (float): accepted maximum value of the ground and
            excited states lattice parameter deviations.
        
        Raises:
            FileNotFoundError: If either the ground_state_file
            or the excited_state_file does not exist.
        """
        if not ground_state_file.exists():
            raise FileNotFoundError("Ground state file not found: {ground_state_file}")
        self.ground_state = read(ground_state_file)

        if not excited_state_file.exists():
            raise FileNotFoundError("Excited state file not found: {excited_state_file}")
        self.excited_state = read(excited_state_file)
        
        if self._is_ground_match_to_excited_state():
            self.eigenvalues = system_optimized_values_obj.eigenvalues
            self.eigenvecotrs = system_optimized_values_obj.eigenvectors
            self.number_of_atoms = system_optimized_values_obj.number_atoms
            self.accepted_cell_deviation = accepted_cell_deviation

    @staticmethod
    def read_poscar(file_path: Path) -> np.ndarray:
        """
        Reads a POSCAR file and returns the structure as an ASE Atoms object.
        
        Args:
            file_path (Path): The path to the POSCAR file.

        Returns:
             np.ndarray: The structure read from the POSCAR file.

        Raises:
            IOError: If there is an issue reading the file.
        """
        if not read(file_path):
            raise IOError(f"Error reading ground state file: {file_path}")
        return read(file_path)

    @staticmethod
    def get_cell(file_path: Path) -> np.ndarray:
        """
        Retrieves the cell parameters from a POSCAR file.

        Args:
            file_path (Path): The path to the POSCAR file.

        Returns:
            np.ndarray: The lattice parameters from the POSCAR file.
        """
        strcuture = PhononSideBand.read_poscar(file_path)
        return strcuture.get_cell()

    @staticmethod
    def get_positions(file_path: Path) -> np.ndarray:
        """
        Retrieves the cell parameters from a POSCAR file.

        Args:
            file_path (Path): The path to the POSCAR file.

        Returns:
            np.ndarray: The atomic positions from the POSCAR file.
        """
        strcuture = PhononSideBand.read_poscar(file_path)
        return strcuture.get_positions()

    @staticmethod
    def get_number_atoms(file_path: Path) -> int:
        """
        Retrieves the number of atoms from a POSCAR file.

        Args:
            file_path (Path): The path to the POSCAR file.

        Returns:
            int: The number of atoms from the POSCAR file.
        """
        strcuture = PhononSideBand.read_poscar(file_path)
        return strcuture.get_number_of_atoms()

    @staticmethod
    def get_chemcial_symbols(file_path: Path) -> list[str]:
        """
        Returns a list of chemical symbols from a POSCAR file.

        Args:
            file_path (Path): The path to the POSCAR file.

        Returns:
            list[str]: Chemical symbols from the POSCAR file.
        """
        strcuture = PhononSideBand.read_poscar(file_path)
        return strcuture.get_chemical_symbols()

    def _check_cell_parameters_match(self):
        cell_ground_state = self.get_cell(self.ground_state)
        cell_excited_state = self.get_cell(self.excited_state)
        cell_difference = np.abs(cell_ground_state - cell_excited_state)
        
        if np.any(cell_difference > self.accepted_cell_deviation):
            raise ValueError('Large deviation between the ground and excited states.')

    def _check_number_of_atoms_match(self):
        ground_state_number_of_atoms = self.get_number_of_atoms(self.ground_state)
        excited_state_number_of_atoms = self.get_number_of_atoms(self.excited_state)
        
        if ground_state_number_of_atoms != excited_state_number_of_atoms:
            raise ValueError('Discrepancy between the number of atoms in ground and excited state structures.')

    def _check_chemical_symbols_match(self):
        ground_state_atoms = self.get_chemical_symbols(self.ground_state)
        excited_state_atoms = self.get_chemical_symbols(self.excited_state)
        
        if ground_state_atoms != excited_state_atoms:
            raise ValueError('Atoms of the ground state are not the same as in the excited state.')

    def _is_ground_match_to_excited_state(self) -> bool:
        """
        Checks if the ground state matches the excited state based on several criteria:
        - Cell parameters within an accepted deviation.
        - Equal number of atoms.
        - Matching chemical symbols.

        Returns:
            bool: True if all criteria are met.

        Raises:
            ValueError: If any criterion is not met.
        """
        self._check_cell_parameters_match()
        self._check_number_of_atoms_match()
        self._check_chemical_symbols_match()

        return True

    def get_displacements(self) -> np.ndarray:
        """
        Returns atomic displacement as a flattened numpy array.
    
        Returns:
        - A numpy array containing the atomic displacements.
        """
        ground_state_positions = PhononSideBand.get_positions(self.ground_state)
        excited_state_positions = PhononSideBand.get_positions(self.excited_state)
        distances = geometry.get_distances(ground_state_positions,
                                           excited_state_positions,
                                           cell=self.ground_state.get_cell(),
                                           pbc=True)
    
        number_atoms = len(self.ground_state.get_positions())
        displacements = [distances[0][i][i] for i in range(number_atoms)]
    
        return np.array(displacements).reshape(-1, 1)

    def get_ground_state_masses(self) -> np.ndarray:
        """
        Extracts atomic masses from a ground state POSCAR file.

        Returns:
        - A numpy array containing the atomic masses.
        """
        
        return self.ground_state.get_masses()

    def normalize_eigenfunction(self, eigenfuntion: np.ndarray) -> np.ndarray:
        """
        Normalizes phonon displacement vectors by atomic masses
        and their magnitudes.

        Parameters:
        - phonons: A numpy array of phonon displacement vectors.
        - masses: A numpy array of atomic masses corresponding
                    to each displacement vector.

        Returns:
        - A numpy array of normalized displacement vectors
            reshaped to a single column.
        """
        reshaped_eigenfunction = np.reshape(eigenfuntion, (-1, 3))
        disp_vec = reshaped_eigenfunction / np.sqrt(self.get_ground_state_masses())
        total_norm = np.sum(np.linalg.norm(disp_vec, axis=1) ** 2)
        normalized_vectors = disp_vec / np.sqrt(total_norm)
    
        return np.reshape(normalized_vectors, (1, -1))


    def get_configurational_coordinates(self, eigenfunction: np.ndarray) -> float:
        """
        Calculates the configurational coordinates from displacement vectors,
        phonon vibrational modes, and atomic masses.

        Parameters:
        - displacements: A numpy array of displacement vectors.
        - phonon_modes: A numpy array representing the vibrational modes' eigenmodes.
        - masses: A numpy array of atomic masses.

        Returns:
        - The first element of the configurational coordinates as a float.
        """
    
        mass_weighted_vib_modes = np.dot(eigenfunction,
                                         self.get_ground_state_masses())
        configurational_coordinates = np.dot(mass_weighted_vib_modes,
                                             self.get_displacements())
    
        return float(configurational_coordinates.ravel()[0])

    @staticmethod
    def get_mode_partial_huang_rhys_factor(
        frequency_thz: float,
        config_coord_sqrt_amu_angstrom: float
        ) -> float:
        """
        Calculates the partial Huang-Rhys factor (HRF) based on
        the vibrational mode's frequency and configuration coordinates.

        Parameters:
        - frequency_thz: The vibrational mode's frequency in terahertz (THz).
        - config_coord_sqrt_amu_angstrom: Square root of the configuration
            coordinate in sqrt(atomic mass units * angstrom^2).
    
        Returns:
        - The partial Huang-Rhys factor (dimensionless).
        """
        angular_freq = const.TERA * utils.get_angular_frequency(frequency_thz)
        conf_coord_pow_2 = np.dot(config_coord_sqrt_amu_angstrom,
                                  config_coord_sqrt_amu_angstrom) * sov.dq2_units
        return 1 / (2 * const.HBAR_TO_EVS) * angular_freq * conf_coord_pow_2

    def get_partial_huang_rhys_factor(self, svmi_obj: SystemVibrationModesInfo,
                                      diag_mass: np.ndarray,
                                      displacements: np.ndarray) -> list[float]:
        """
        Calculates partial Huang-Rhys factors for given phonon information,
        atomic masses, diagonalized mass matrix, and displacements.
    
        Parameters:
        - svmi_obj: An instance of SystemVibrationModesInfo containing phonon information.
        - masses: A numpy array of atomic masses.
        - diag_mass: A numpy array representing the diagonalized mass matrix.
        - displacements: A numpy array of atomic displacements.
    
        Returns:
        - A list of partial Huang-Rhys factors.
        """
        huang_rhys_factor = [0.0] * (3 * svmi_obj.number_atoms)
        configuration_coordinates = []
        for mode in range(3 * svmi_obj.number_atoms):
            normal_phon_mode = self.normalize_eigenfunction(
                np.array(svmi_obj.eigenvectors[mode]))
            conf_coord_mode = self.get_configurational_coordinates(normal_phon_mode) #diag_mass)
            configuration_coordinates.append(conf_coord_mode ** 2)
            huang_rhys_factor[mode] = PhononSideBand.get_mode_partial_huang_rhys_factor(svmi_obj.eigenvalues[mode],
                                                                             conf_coord_mode)

        #apply acoustic rule    
        huang_rhys_factor[0:3] = [0.0, 0.0, 0.0]
        return huang_rhys_factor

    def spectral_function(partial_huang_rhys_factor: list,
                          number_grids: int,
                          photon_energy: np.ndarray,
                          vibibration_eigenvalues: np.ndarray) -> np.ndarray:
        """
        Calculates the spectral function using partial Huang-Rhys factors, energy grids, and vibrational mode energies.

        Parameters:
        - partialhrfs: List of partial Huang-Rhys factors.
        - n_grids: Number of energy grids.
        - energy: Numpy array representing the phonon energy range.
        - vib_mode_energy: Numpy array of phonon vibrational modes energies in eV units.

        Returns:
        - A numpy array representing the spectral function across the energy grid.
        """
        spectral_huang_rhys_factor = np.zeros(number_grids)
        for index, value in enumerate(partial_huang_rhys_factor):
            spectral_huang_rhys_factor += value * utils.fit_gaussian_functions(photon_energy,
                                                                               vibibration_eigenvalues[index],
                                                                               sigma=0.005)
        return spectral_huang_rhys_factor

    def generate_time_dependent_function(total_time: np.ndarray,
                                         energy_levels: np.ndarray,
                                         energy_interval: float,
                                         spectral_function: np.ndarray,
                                         temperature: float) -> np.ndarray:
        """
        Calculate the time-dependent generating function for a given set of parameters.

        Parameters:
        - total_time (np.ndarray): An array of time points.
        - energy_levels (np.ndarray): An array of energy levels.
        - energy_interval (float): The energy interval (dE).
        - spectral_function (np.ndarray): The spectral function corresponding to each energy level.
        - temperature (float): The temperature.

        Returns:
        - np.ndarray: The time-dependent generating function as a numpy array.
        """

        spectral_function = np.zeros(len(total_time), dtype=complex)
        for j, time_point in enumerate(total_time):
            for i, energy in enumerate(energy_levels):
                temperature_factor = utils.calculate_boltzmann_distribution(energy, temperature)
                exponential_factor = np.exp(1j * energy * time_point * const.PICO / const.HBAR_TO_EVS)
            
                positive_term = spectral_function[i] * temperature_factor * exponential_factor
                negative_term = spectral_function[i] * (1 + temperature_factor) * np.exp(-exponential_factor.imag)
   
                spectral_function[j] += (positive_term + negative_term) * energy_interval
            
        return np.exp(spectral_function - spectral_function[0])

    def calculate_optical_absorption_spectrum(generating_function: np.ndarray,
                                              time_period: float,
                                              zero_phonon_line_energy: float) -> np.ndarray:
        """
        Calculate the optical absorption spectrum from a given generating function.

        Parameters:
        - generating_function (np.ndarray): The generating function as a numpy array.
        - time_period (float): The time period over which the function is defined.
        - zero_phonon_line_energy (float): The energy of the zero-phonon line (ZPL).

        Returns:
        - np.ndarray: The scaled optical absorption spectrum.
        """
        optical_absorption_spectrum = np.fft.ifft(generating_function) / (2 * np.pi)

        photon_angular_frequency = (2 * np.pi / time_period / const.PICO) * np.arange(len(generating_function))
        photon_energy = const.HBAR_TO_EVS * photon_angular_frequency

        scaled_absorption_spectrum = (((zero_phonon_line_energy - photon_energy) / const.HBAR_TO_EVS) ** 3) * np.abs(optical_absorption_spectrum)

        return scaled_absorption_spectrum

obj = PhononSideBand(ground_state_file=Path('POSCAR-gs'), excited_state_file=Path('POSCAR-es'))
print(obj.get_ground_state_masses().shape)