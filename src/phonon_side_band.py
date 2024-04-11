from pathlib import Path
import numpy as np
from ase import geometry
from ase.io import read

import constants as const
import utils
import time

# from parse_valid_yaml import SystemVibrationModesInfo
# from parse_valid_yaml import ParsePhonopyYamlFile

from parse_valid_outcar import SystemVibrationModesInfo
from parse_valid_outcar import ParseOutcarFile

class PhononSideBand:
    """
    This class computes optical absorption phonon side band.
    """

    def __init__(
        self,
        ground_state_file: Path,
        excited_state_file: Path,
        zero_phonon_line: float,
        system_vibration_data: SystemVibrationModesInfo,
        accepted_cell_deviation: float = 0.1,
    ) -> None:
        """
        Initializes PhononSideBand instance.

        Args:
        - ground_state_file (Path): Path to the ground state POSCAR file.
        - excited_state_file (Path): Path to the excited state POSCAR file.
        - zero_phonon_line (float): Zero Phonon Line energy in eV.
        - system_optimized_values_obj (SystemVibrationModesInfo): Stores vibration
            modes information including eigenvectors, eigenvalues, and number of atoms.
        - accepted_cell_deviation (float): accepted maximum value of the ground and
            excited states lattice parameter deviations.

        Raises:
            FileNotFoundError: If either the ground_state_file
            or the excited_state_file does not exist.
            ValueError: If zero_phonon_line is negative.
        """
        if not ground_state_file.exists():
            raise FileNotFoundError("Ground state file not found: {ground_state_file}!")
        self.ground_state = ground_state_file

        if not excited_state_file.exists():
            raise FileNotFoundError(
                "Excited state file not found: {excited_state_file}!"
            )
        self.excited_state = excited_state_file

        if zero_phonon_line < 0:
            raise ValueError("ZPL must be positive!")
        self.zero_phonon_line = zero_phonon_line

        self.eigenvalues = system_vibration_data.eigenvalues
        self.eigenvectors = system_vibration_data.eigenvectors
        self.number_of_atoms = system_vibration_data.number_atoms
        self.accepted_cell_deviation = accepted_cell_deviation

    def _check_cell_parameters_match(self):
        cell_ground_state = utils.get_cell(self.ground_state)
        cell_excited_state = utils.get_cell(self.excited_state)
        cell_difference = np.abs(cell_ground_state - cell_excited_state)

        if np.any(cell_difference > self.accepted_cell_deviation):
            raise ValueError("Cell deviation between the ground and excited states!")

    def _check_number_of_atoms_match(self):
        ground_state_number_of_atoms = utils.get_number_atoms(self.ground_state)
        excited_state_number_of_atoms = utils.get_number_atoms(self.excited_state)

        if ground_state_number_of_atoms != excited_state_number_of_atoms:
            raise ValueError("Ground and excited states' number of atoms NOT match!")

    def _check_chemical_symbols_match(self):
        ground_state_atoms = utils.get_chemical_symbols(self.ground_state)
        excited_state_atoms = utils.get_chemical_symbols(self.excited_state)

        if ground_state_atoms != excited_state_atoms:
            raise ValueError("Atoms of the ground and excited states are NOT similar!")

    def _is_ground_match_to_excited_state(self) -> bool:
        """
        Checks if the ground state matches the excited state based on:
        - Cell parameters within an accepted deviation.
        - Equal number of atoms.
        - Matching chemical symbols.

        Returns:
        - bool: True if all criteria are met.

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
        self._is_ground_match_to_excited_state()

        ground_state_positions = utils.get_positions(self.ground_state)
        excited_state_positions = utils.get_positions(self.excited_state)
        ground_state_cell = utils.get_cell(self.ground_state)

        distances = geometry.get_distances(
            ground_state_positions,
            excited_state_positions,
            cell=ground_state_cell,
            pbc=True,
        )

        number_atoms = utils.get_number_atoms(self.ground_state)
        displacements_list = [
            element for i in range(number_atoms) for element in distances[0][i][i]
        ]

        return np.array(displacements_list)

    def get_ground_state_masses(self) -> np.ndarray:
        """
        Extracts atomic masses from a ground state POSCAR file.

        Returns:
        - A numpy array containing the atomic masses.
        """
        ground_state_strcuture = read(self.ground_state)
        return ground_state_strcuture.get_masses()

    def _diagonalize_masses_matrix(self):
        """
        Diagonalizes mass (1,N) matrix.

        Returns:
        - np.ndarray: (N,N) mass array.
        """
        repeated_masses = np.repeat(self.get_ground_state_masses(), 3)
        return np.diag(repeated_masses)

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
        repeated_masses = np.repeat(self.get_ground_state_masses(), 3)
        weighted_eigenfunction = eigenfuntion / np.sqrt(repeated_masses)
        norm_value_square = np.linalg.norm(weighted_eigenfunction) ** 2

        return weighted_eigenfunction / np.sqrt(norm_value_square)

    def _get_mode_configuration_coordinates(
        self,
        disp_vector: np.ndarray,
        sqrt_mass_matrix: np.ndarray,
        eigenfunction: np.ndarray,
    ) -> float:
        """
        Calculates the configurational coordinates from displacement vectors,
        phonon vibrational modes, and square root of atomic masses.

        Args:
        - disp_vector (np.ndarray): an array of atomic displacement with shape(-1, 1)
        - sqrt_mass_matrix (np.ndarray): a numpy array of square root of atomi masses
        - eigenfunction (np.ndarray): a numpy array representing the vibrational
            eigenfunction, expected shape (N,).

        Returns:
        - float: The configuration coordinate corresponding to the eigenfunction,
            derived from the inner product of normalized eigenfunction,
            diagonalized mass matrix, and displacement vectors.
        """
        norm_eigenfunc = self.normalize_eigenfunction(eigenfunction).reshape(1, -1)

        return float((norm_eigenfunc @ (sqrt_mass_matrix @ disp_vector)).ravel()[0])

    def get_configuration_coordinates(self) -> list[float]:
        """
        Calculates configuration coordinates for all vibrational modes.

        Returns:
        - list[float]: a list of configuration coordinates corresponding all modes.
        """
        number_of_modes = 3 * utils.get_number_atoms(self.ground_state)
        disp_vector = self.get_displacements().reshape(-1, 1)
        sqrt_mass_matrix = np.sqrt(self._diagonalize_masses_matrix())

        configuration_coordinates = [
            self._get_mode_configuration_coordinates(
                disp_vector, sqrt_mass_matrix, self.eigenvectors[mode]
            )
            for mode in range(number_of_modes)
        ]

        # TODO: save data to a text file
        return configuration_coordinates

    def _get_mode_partial_huang_rhys_factor(
        self,
        disp_vector: np.ndarray,
        sqrt_mass_matrix: np.ndarray,
        eigenvalue: float,
        eigenfunction: np.ndarray,
    ) -> float:
        """
        Calculates the partial Huang-Rhys factor (HRF) based on
        the vibrational mode's frequency and configuration coordinates.

        Args:
        - disp_vector (np.ndarray): an array of atomic displacement with shape(-1, 1)
        - sqrt_mass_matrix (np.ndarray): a numpy array of square root of atomi masses
        - eigenvector (np.ndarray): A numpy array representing the vibration eigenfunction.
        - eigenvalue (float): The frequency of the vibrational mode.

        Returns:
        - float: The partial Huang-Rhys factor (dimensionless) for the specified vibrational mode.
        """
        angular_freq = utils.get_angular_frequency(eigenvalue)
        configuration_coordinate = self._get_mode_configuration_coordinates(
            disp_vector, sqrt_mass_matrix, eigenfunction
        )

        unit_conversion = (const.ANGSTROM**2) * const.AMU_TO_KG * const.JOULE_TO_EV
        configuration_coordinate_squared = (
            configuration_coordinate**2
        ) * unit_conversion

        return (angular_freq * configuration_coordinate_squared) / (
            2 * const.HBAR_TO_EVS
        )

    def get_partial_huang_rhys_factor(self) -> list[float]:
        """
        Calculates partial Huang-Rhys factors for all vibrational modes.

        Returns:
        - list[float]: a list of partial Huang-Rhys factors for each mode,
        with the acoustic modes set to zero by the acoustic rule.
        """
        number_of_modes = 3 * utils.get_number_atoms(self.ground_state)
        disp_vector = self.get_displacements().reshape(-1, 1)
        sqrt_mass_matrix = np.sqrt(self._diagonalize_masses_matrix())

        huang_rhys_factor = [
            self._get_mode_partial_huang_rhys_factor(
                disp_vector,
                sqrt_mass_matrix,
                self.eigenvalues[mode],
                self.eigenvectors[mode],
            )
            for mode in range(number_of_modes)
        ]

        # acoustic rule
        huang_rhys_factor[:3] = [0.0] * 3

        # TODO: save data to a text file
        return huang_rhys_factor

    def _get_phonon_energy_grids(self, number_grids=1000, buffer=3):
        """
        Generates an array of energy grid points based on the phonon eigenvalues.

        Args:
        - number_grids (int): The number of points in the energy grid. Defaults to 500.
        - buffer (float): An additional energy buffer to extend beyond the maximum eigenvalue.
            Defaults to 2.

        Returns:
            np.ndarray: An array of linearly spaced energy grid points.
        """
        unit_conversion = const.THz_TO_eV
        
        if number_grids <= 0:
            raise ValueError("number_grids must be a positive integer")
        max_eigenvalue = (np.max(self.eigenvalues) + buffer) * unit_conversion
        return np.linspace(0.001, max_eigenvalue, number_grids)

    def huang_rhys_factor_spectral_function(
        self, number_grids=1000, sigma=0.003
    ) -> np.ndarray:
        """
        Calculates the spectral function by fitting a Gaussian function on partial Huang-Rhys
        factors corresponding to phonon vibrational mode energies (eigenvalues).

        Args:
        - number_grids (int): Number of energy grid points. Defaults to 500.

        Returns:
            np.ndarray: A numpy array representing the spectral function across the energy grid.
        """
        unit_conversion = const.THz_TO_eV

        energy_grids = self._get_phonon_energy_grids(number_grids)
        spectral_function = np.zeros(number_grids)

        partial_factors = self.get_partial_huang_rhys_factor()
        for index, hrf_value in enumerate(partial_factors):
            spectral_function += hrf_value * np.vectorize(utils.fit_gaussian_functions)(
                energy_grids, unit_conversion * self.eigenvalues[index], sigma=sigma
            )

        # TODO: save data to a text file

        return spectral_function

    def generate_time_dependent_function(
        self, temperature: float, number_grids=1000, time_period=1
    ) -> np.ndarray:
        """
        Calculate the time-dependent generating function for a given set of parameters.
        """

        unit_conversion = const.PICO / const.HBAR_TO_EVS

        energy_grids = self._get_phonon_energy_grids(number_grids=number_grids)
        dE = energy_grids[1] - energy_grids[0]
        hrf_spectral_function = self.huang_rhys_factor_spectral_function(
            number_grids=number_grids
        )

        #dE = self.zero_phonon_line / number_grids

        time_grids = np.linspace(-time_period, time_period, number_grids)
        time_dependent_spectral_function = np.zeros((number_grids,), dtype=complex)

        for j in range(number_grids):
            for i in range(number_grids):
                temperature_factor = utils.calculate_boltzmann_distribution(
                    energy_grids[i], temperature
                )

                positive_term = (
                    hrf_spectral_function[i]
                    * temperature_factor
                    * np.exp(1j * energy_grids[i] * time_grids[j] * unit_conversion)
                )
                negative_term = (
                    hrf_spectral_function[i]
                    * (1 + temperature_factor)
                    * np.exp(-1j * energy_grids[i] * time_grids[j] * unit_conversion)
                )

                time_dependent_spectral_function[j] += (
                    positive_term + negative_term
                ) * dE

        return np.exp(time_dependent_spectral_function - time_dependent_spectral_function[0])

    def calculate_optical_absorption_spectrum(
        self,
        temperature: float,
        number_grids=1000,
        time_period=1,
    ) -> np.ndarray:
        """
        Calculate the optical absorption spectrum from a given generating function.

        Parameters:
        - generating_function (np.ndarray): The generating function as a numpy array.
        - time_period (float): The time period over which the function is defined.
        - zero_phonon_line_energy (float): The energy of the zero-phonon line (ZPL).

        Returns:
        - np.ndarray: The scaled optical absorption spectrum.
        """
        gen_func = self.generate_time_dependent_function(temperature=temperature)
        time_grids = np.linspace(-time_period, time_period, number_grids)

        optical_absorption_spectrum = np.fft.ifft(gen_func) / (2 * np.pi)

        photon_angular_frequency = (2 * np.pi / time_period / const.PICO) * np.arange(
            number_grids
        )
        photon_energy = const.HBAR_TO_EVS * photon_angular_frequency

        scaled_absorption_spectrum = (
            ((self.zero_phonon_line - photon_energy) / const.HBAR_TO_EVS) ** 3
        ) * np.abs(optical_absorption_spectrum)
        
        import matplotlib.pyplot as plt
        plt.plot(photon_energy,
                 scaled_absorption_spectrum,
                 color='red',
                 marker='o',
                 markersize=1)
        # plt.plot(time_grids, gen_func.real, 'b--')
        plt.show()
        
        return scaled_absorption_spectrum


import time

start_time = time.time()

# obj1 = ParsePhonopyYamlFile(Path("qpoints.yaml"))
# # print(obj1.get_vibration_data().eigenvectors)

obj1 = ParseOutcarFile(Path("OUTCAR-ph.vasp"))

obj = PhononSideBand(
    ground_state_file=Path("CONTCAR-gs.vasp"),
    excited_state_file=Path("CONTCAR-ex-up.vasp"),
    zero_phonon_line=1.1,
    system_vibration_data=obj1.get_vibration_data(),
    accepted_cell_deviation=0.1,
)


obj.calculate_optical_absorption_spectrum(temperature=30)
#obj.huang_rhys_factor_spectral_function()

end_time = time.time()
runtime = end_time - start_time
print(f"Runtime: {runtime} seconds")

# print(np.sum(obj.get_partial_huang_rhys_factor()))
