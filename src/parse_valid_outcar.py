
from typing import Optional
from pathlib import Path
import numpy as np

import time

class SystemVibrationModesInfo:
    """
    Stores and provides access to the vibrational modes information of a system.
    This class encapsulates eigenvectors, eigenvalues, and number of atoms
    associated with the vibrational modes of a crystalline system.

    Args:
    - eigenvectors (np.ndarray): an array of vibrational eigenvectors.
    - eigenvalues (np.ndarray): an array of vibrational frequencies.
    - number_atoms (int): the number of atoms in the system.
    """

    def __init__(
        self, eigenvectors: np.array, eigenvalues: np.array, number_atoms: int
    ):
        """
        Initializes an instance of SystemVibrationModesInfo.

        Args:
        - eigenvectors (np.ndarray): an array of vibrational mode's eigenvectors.
        - eigenvalues (np.ndarray): an array of vibrational mode's energies.
            Should align with the `eigenvectors` array.
        - number_atoms (int): the number of atoms in the system.

        Raises:
            ValueError: If the lengths of `eigenvectors` and `eigenvalues` do not match.
        """
        if len(eigenvectors) != len(eigenvalues):
            raise ValueError(
                "Length of eigenvectors does not match number of eigenvalues."
            )

        self.eigenvectors = eigenvectors
        self.eigenvalues = eigenvalues
        self.number_atoms = number_atoms

class ParseOutcarFile:
    """summay to be written
    Args:
    
    Methods
    """
    def __init__(self, outcar_file: Path) -> None:
        """
        Initializes an instance of ParseOutcarFile.

        Args:
        - outcar_file (Path): path to OUTCAR file.

        Raises:
            FileNotFoundError: If file not found in the given path.
        """
        if not outcar_file.exists():
            raise FileNotFoundError(f"File {outcar_file} does not exist!")
        self.outcar_file = outcar_file

    @staticmethod
    def _extract_displacement(line) -> list[float]:
        """Extracts and returns the atomic displacement from a line."""
        displacement_components = line.split()[3:6]
        return [float(component) for component in displacement_components]

    @staticmethod
    def _extract_frequency(line) -> float:
        """Extracts and returns the frequency from a line."""
        return float(line.split()[-8])
    
    def get_atomic_displacements(self) -> Optional[list[list[float]]]:
        """Reads the OUTCAR file and extracts atomic displacements following frequency lines."""
        displacements = []
        try:
            with open(self.outcar_file, 'r') as phon_file:
                for line in phon_file:
                    if " f  =  " in line or "f/i=" in line:
                        # Skip one line immediately following the frequency line.
                        next(phon_file)

                        # Process the following lines for displacements.
                        for displacement_line in phon_file:
                            if len(displacement_line.split()) == 6:
                                displacements.append(self.__class__._extract_displacement(displacement_line))
                            else:
                                # Exit the inner loop if the line format does not match.
                                break
        except Exception as e:
            print(f"An error occurred while reading the file: {e}. Does it include vibration analysis?")

        return displacements

    def get_vibration_frequencies(self) -> Optional[list[float]]:
        """Reads the OUTCAR file and extracts vibration frequencies."""
        frequencies = []
        try:
            with open(self.outcar_file, 'r') as phon_file:
                for line in phon_file:
                    if " f  =  " in line or "f/i=" in line:
                        frequencies.append(self.__class__._extract_frequency(line))
        except Exception as e:
            print(f"An error occurred while reading the file: {e}. Does it include vibration analysis?")

        return frequencies

    def get_number_of_atoms(self) -> Optional[int]:
        """Extracts the total number of atoms from a VASP OUTCAR file.
        
        This method opens the specified OUTCAR file, searches for the line that contains 
        'NIONS', and extracts the number of atoms from it. The number of atoms is 
        expected to be the last integer value on the line containing 'NIONS'.
        
        Returns:
            The total number of atoms as an integer if found, otherwise None.
        """
        try:
            with open(self.outcar_file, 'r') as file:
                for line in file:
                    if "NIONS" in line:
                        # Extract the number of atoms and return it.
                        number_of_atoms = int(line.split()[-1])
                        return number_of_atoms

        except Exception as e:
            print(f"An error occurred while reading the file: {e}")
        return None

    def _is_outcar_valid(self) -> bool:
        """
        Validates the structure and essential keys of an OUTCAR file.

        This method checks whether the OUTCAR file contains the required keys by verifying
        the consistency between the number of atoms and the number of vibrational modes.

        Returns:
        - bool: True if the OUTCAR file meets the required validation criteria.

        Raises:
            ValueError: If the number of vibrational frequencies does not match three times
            the number of atoms, or if any atomic displacement mode does not correspond to 
            three times the number of atoms, indicating a mismatch between expected and actual data.
        """
        total_atoms = self.get_number_of_atoms()
        vibration_eigenvalues = self.get_vibration_frequencies()
        vibration_eigenfuntions = self.get_atomic_displacements()
        
        number_of_modes = 3 * total_atoms
        
        if not len(vibration_eigenvalues) == number_of_modes:
            raise ValueError('Mismatch between the number of atoms and the number of vibrational frequencies.')

        if not len(vibration_eigenfuntions) == number_of_modes * total_atoms:
            raise ValueError('Mismatch in the atomic displacements for one or more vibrational modes.')
        
        return True


    def get_vibration_data(self) -> SystemVibrationModesInfo:
        """
        Creates a SystemVibrationModesInfo object if the given OUTCAR file is valid.

        Returns:
        - SystemVibrationModesInfo: an object populated with vibration mode data.
        """

        self._is_outcar_valid()

        frequency_values = self.get_vibration_frequencies()
        eigenvectors = self.get_atomic_displacements()
        number_atoms = self.get_number_of_atoms()

        eigenvalues = np.array(frequency_values, dtype=float)
        reshaped_eigenvectors = np.reshape(eigenvectors, (3 * number_atoms, -1))

        return SystemVibrationModesInfo(
            eigenvectors=reshaped_eigenvectors,
            eigenvalues=eigenvalues,
            number_atoms=number_atoms,
        )