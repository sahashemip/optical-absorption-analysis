"""
This script contains several utilities as follows:

Methods:
- fit_gaussian_functions
- get_angular_frequency
- calculate_boltzmann_distribution
- write_data_to_file
"""

import csv
from pathlib import Path
from typing import Iterable, Union
from ase.io import read
import numpy as np
import constants as const

def fit_gaussian_functions(x: float, mean: float, sigma: float) -> float:
    """
    Fits Gaussian function on a given x, mean, and standard deviation.
    
    Args:
    - x: the point at which the Gaussian function is evaluated.
    - mean: the mean of the Gaussian distribution.
    - sigma: the standard deviation of the Gaussian distribution; must be positive.
    
    Returns:
    - The value of the Gaussian function at x.
    
    Raises:
        If sigma is negative, raises ValuError.
    """
    if sigma <= 0.0:
        raise ValueError(f"Sigma value must be positive, but it is {sigma}!") 
    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mean) / sigma) ** 2)

def get_angular_frequency(frequency_thz: float) -> float:
    """
    Converts a frequency from terahertz (THz) to
    angular frequency (radians per second).
    
    Parameters:
    - frequency_thz: Frequency in terahertz (THz).
    
    Returns:
    - Angular frequency in radians per second.
    """
    return 2 * const.TERA * np.pi * frequency_thz

def calculate_boltzmann_distribution(energy: float, temperature: float) -> float:
    """
    Calculate the Boltzmann distribution for a given energy and temperature.

    Args:
    - energy (float): The energy level in electron volts (eV).
    - temperature (float): The temperature in Kelvin (K) and greater than 0.

    Returns:
    - float: The Boltzmann distribution for the given energy and temperature.

    Raises:
        ValueError: If temperature is negative.
    """

    if temperature <= 0.0:
        raise ValueError(f"Temperature must be positive!") 
    return 1 / (np.exp(energy / (const.KB * temperature)) - 1)

def read_poscar(file_path: Path) -> np.ndarray:
    """
    Reads a POSCAR file and returns the structure as an ASE Atoms object.

    Args:
    - file_path (Path): The path to the POSCAR file.

    Returns:
    - np.ndarray: The structure read from the POSCAR file.

    Raises:
        IOError: If there is an issue reading the file.
    """
    if not read(file_path):
        raise IOError(f"Error reading ground state file: {file_path}")
    return read(file_path)

def get_cell(file_path: Path) -> np.ndarray:
    """
    Retrieves the cell parameters from a POSCAR file.

    Args:
    - file_path (Path): The path to the POSCAR file.

    Returns:
    - np.ndarray: The lattice parameters from the POSCAR file.
    """
    strcuture = read_poscar(file_path)
    return strcuture.get_cell()

def get_positions(file_path: Path) -> np.ndarray:
    """
    Retrieves atomic positions from a POSCAR file.

    Args:
    - file_path (Path): The path to the POSCAR file.

    Returns:
    - np.ndarray: The atomic positions from the POSCAR file.
    """
    strcuture = read_poscar(file_path)
    return strcuture.get_positions()

def get_number_atoms(file_path: Path) -> int:
    """
    Retrieves the number of atoms from a POSCAR file.

    Args:
    - file_path (Path): The path to the POSCAR file.

    Returns:
    - int: The number of atoms from the POSCAR file.
    """
    strcuture = read_poscar(file_path)
    return strcuture.get_global_number_of_atoms()

def get_chemical_symbols(file_path: Path) -> list[str]:
    """
    Returns a list of chemical symbols from a POSCAR file.

    Args:
    - file_path (Path): The path to the POSCAR file.

    Returns:
    - list[str]: Chemical symbols from the POSCAR file.
    """
    strcuture = read_poscar(file_path)
    return strcuture.get_chemical_symbols()

def write_data_to_file(filename: Path,
                       x_variable: Iterable[Union[float, int]],
                       y_variable: Iterable[Union[float, int]]) -> None:
    """
    Writes pairs of values to a tab-delimited text format.

    Args:
    - filename (Path): The path to the file to write into.
    - x_variable (Iterable[Union[float, int]]): An iterable of x values.
    - y_variable (Iterable[Union[float, int]]): An iterable of y values,
        must be the same length as x.

    Raises:
        ValueError: If x and y do not have the same length.
    """
    if len(x_variable) != len(y_variable):
        raise ValueError("The length of x and y must be the same.")
        
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerows(zip(x_variable, y_variable))
