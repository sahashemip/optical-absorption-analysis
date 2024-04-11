from typing import Optional
from pathlib import Path
import numpy as np


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


class ParsePhonopyYamlFile:
    """
    Parses Phonopy generated YAML file that conatins the vibrational modes' information of
    only the gamma-point.

    This class extracts vibrational mode's eigenvector, eigenvalues and the number of
    atoms. It also check YAML file structure correctness.

    Args:
    - yamlfile (Path): path to YAML file.

    Methods:
    - _extract_vibration_eigenvectors
    - _extract_yaml_values_by_key
    - _is_yaml_valid
    - get_vibration_data
    """

    def __init__(self, yamlfile: Path) -> None:
        """
        Initializes an instance of ParsePhonopyYamlFile.

        Args:
        - yamlfile (Path): path to YAML file.

        Raises:
            FileNotFoundError: If file not found in the given path.
        """
        if not yamlfile.exists():
            raise FileNotFoundError(f"File {yamlfile} does not exist!")
        self.yamlfile = yamlfile

    def _extract_yaml_values_by_key(self, key: str) -> Optional[list[float]]:
        """
        Extracts values related to a specified 'key' from the given YAML file.

        Args:
        - key (str): Key parameter for a dictionary.

        Returns:
        - A list of extracted values as floats, if found; otherwise, None.
        """
        list_of_values = []
        with open(self.yamlfile, "r") as file:
            for line in file:
                if key in line:
                    list_of_values.append(float(line.split()[-1]))
            if list_of_values:
                return list_of_values
        return None

    def _extract_vibration_eigenvectors(self) -> Optional[list[list[float]]]:
        """
        Extracts vibrational eigenvectors from the given YAML file.

        Returns:
        - A nested list of extracted eigenvector values as floats,
            if found; otherwise, None.
        """
        eigenvectors = []
        with open(self.yamlfile, "r") as file:
            for line in file:
                if "- # atom " in line:
                    atom_displacement = []
                    for _ in range(3):
                        split_line = next(file).split()[2]
                        atom_displacement.append(float(split_line[:-1]))
                    eigenvectors.append(atom_displacement)
            if eigenvectors:
                return eigenvectors
        return None

    def _is_yaml_valid(self) -> bool:
        """
        Validates the structure and essential keys of a given YAML file
        obtained from Phonopy analysis.

        Returns:
        - bool: True if the YAML file's format and required keys are present,
            otherwise raises ValueError.

        Raises:
            ValueError: If essential keys are missing
            or
            if there are multiple entries for a single key where only one is expected.
        """
        required_keys = ["nqpoint:", "natom:  ", "frequency: "]
        values = {key: self._extract_yaml_values_by_key(key) for key in required_keys}
        eigenvectors = self._extract_vibration_eigenvectors()

        for key, value in values.items():
            if value is None:
                raise ValueError(f"Key '{key}' is missing in the YAML file.")
            if key in ["nqpoint:", "natom:  "] and len(value) != 1:
                raise ValueError(f"Invalid YAML file! Only one '{key}' must exist.")

        if eigenvectors is None:
            raise ValueError("Eigenvectors are missing in the YAML file.")
        return True

    def get_vibration_data(self) -> SystemVibrationModesInfo:
        """
        Creates a SystemVibrationModesInfo object if the given YAML file is valid.

        Returns:
        - SystemVibrationModesInfo: an object populated with vibration mode data.
        """

        self._is_yaml_valid()

        number_atoms_values = self._extract_yaml_values_by_key("natom:  ")
        frequency_values = self._extract_yaml_values_by_key("frequency: ")
        eigenvectors = self._extract_vibration_eigenvectors()

        number_atoms = int(number_atoms_values[0])
        eigenvalues = np.array(frequency_values, dtype=float)
        reshaped_eigenvectors = np.reshape(eigenvectors, (3 * number_atoms, -1))

        return SystemVibrationModesInfo(
            eigenvectors=reshaped_eigenvectors,
            eigenvalues=eigenvalues,
            number_atoms=number_atoms,
        )

