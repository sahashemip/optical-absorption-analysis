from typing import Optional
from pathlib import Path
import numpy as np


class SystemVibrationModesInfo:
    """
    Stores and provides access to the vibrational modesinformation of a system.
    This class encapsulates the eigenvectors, eigenvalues, and the total
    number of atoms associated with the vibrational modes of a crystalline system.

    Attributes:
        eigenvectors (np.ndarray): An array of eigenvectors associated with the vibrational
            modes of the system. Each eigenvector represents a vibrational mode.
        eigenvalues (np.ndarray): An array of eigenvalues corresponding to the vibrational
            frequencies of the modes represented by the eigenvectors.
        number_atoms (int): The total number of atoms in the system.
    """
    
    def __init__(self,
                 eigenvectors: np.array,
                 eigenvalues: np.array,
                 number_atoms: int):
        """Returns the vibrational modes related information."""
        self.eigenvectors = eigenvectors
        self.eigenvalues = eigenvalues
        self.number_atoms = number_atoms


class ParsePhonopyYamlFile:
    """
    Extracts abd
    
    Methods:
        - extract_vibration_eigenvectors
        - extract_yaml_values_by_key
        - set_system_vib_mode_info
        - is_yaml_valid
    """
    
    def __init__(self, yamlfile: Path) -> None:
        self.yamlfile = yamlfile
        
        if not self.file_path.exists():
            raise FileNotFoundError(
                f"The file {self.yamlfile} does not exist."
                )

    def extract_yaml_values_by_key(self, key: str) -> Optional[list[float]]:
        """
        Extracts values associated with a specified 'key' from a given YAML file.

        Args:
            key (str): Key parameter for a dictionary.
        
        Returns:
            Optional[List[float]]: A list of extracted values as floats,
            if found; otherwise, None.
        """
        list_of_values = []
        with open(self.yamlfile, 'r') as file:
            for line in file:
                if key in line:
                    list_of_values.append(float(line.split()[-1]))
            if list_of_values:
                return list_of_values
        return None

    def extract_vibration_eigenvectors(self) -> Optional[list[list[float]]]:
        """
        Extracts vibration eigenvectors from a given YAML file.
    
        Returns:
            Optional[List[float]]: A nested list of extracted eigenvector values as floats,
            if found; otherwise, None.
        """
        eigenvectors = []
        with open(self.yamlfile, 'r') as file:
            for line in file:
                if '- # atom ' in line:
                    atom_displacement = []
                    for _ in range(3):
                        split_line = next(file).split()[2]
                        atom_displacement.append(float(split_line[:-1]))
                    eigenvectors.append(atom_displacement)
            if eigenvectors:
                return eigenvectors
        return None

    def set_system_vib_mode_info(self) -> SystemVibrationModesInfo:
        """
        Creates a SystemVibrationModesInfo object from a YAML file.

        Returns:
            SystemVibrationModesInfo: An object populated with vibration mode data.
        """
    
        number_atoms_values = self.extract_yaml_values_by_key('natom:  ')
        frequency_values = self.extract_yaml_values_by_key('frequency: ')
        eigenvectors = self.extract_vibration_eigenvectors()

        number_atoms = int(number_atoms_values[0])
        eigenvalues = np.array(frequency_values, dtype=float)
        reshaped_eigenvectors = np.reshape(eigenvectors, (3 * number_atoms, -1))
    
        return SystemVibrationModesInfo(eigenvectors=reshaped_eigenvectors,
                                        eigenvalues=eigenvalues,
                                        number_atoms=number_atoms)

    def is_yaml_valid(self) -> bool:
        """
        Validates the structure and essential keys of a given YAML file
        obtained from Phonopy analysis.

        Returns:
            bool: True if the YAML file's format and required keys are present,
            otherwise raises ValueError.

        Raises:
            ValueError: If essential keys are missing
            or
            if there are multiple entries for a single key where only one is expected.
        """
        required_keys = ['nqpoint:', 'natom:  ', 'frequency: ']
        values = {key: self.extract_yaml_values_by_key(key) for key in required_keys}
        eigenvectors = self.extract_vibration_eigenvectors()

        for key, value in values.items():
            if value is None:
                raise ValueError(f"Key '{key}' is missing in the YAML file.")
            if key in ['nqpoint:', 'natom:  '] and len(value) != 1:
                raise ValueError(f"Invalid YAML file! Only one '{key}' must exist.")

        if eigenvectors is None:
            raise ValueError("Eigenvectors are missing in the YAML file.")
        return True