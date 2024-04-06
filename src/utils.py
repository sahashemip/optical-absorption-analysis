import csv
from pathlib import Path
from typing import Iterable, Union

import numpy as np

import constants as const

def fit_gaussian_functions(x: float, mean: float, sigma: float) -> float:
    """
    Calculates the Gaussian function value for a given x, mean, and standard deviation.
    
    The Gaussian function is defined as:
    f(x) = (1 / (sigma * sqrt(2 * pi))) * exp(- (x - mean)^2 / (2 * sigma^2))
    
    Parameters:
    - x: The point at which the Gaussian function is evaluated.
    - mean: The mean (mu) of the Gaussian distribution.
    - sigma: The standard deviation (sigma) of the Gaussian distribution; must be positive.
    
    Returns:
    - The value of the Gaussian function at x.
    """
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
    return 2 * np.pi * frequency_thz

def write_data_to_file(filename: Path,
                       x_variable: Iterable[Union[float, int]],
                       y_variable: Iterable[Union[float, int]]) -> None:
    """
    Writes pairs of x and y values to a specified text file in a tab-delimited format.

    Parameters:
    - filename (Path): The path to the file to write into.
    - x (Iterable[Union[float, int]]): An iterable of x values.
    - y (Iterable[Union[float, int]]): An iterable of y values, must be the same length as x.

    Raises:
    - ValueError: If x and y do not have the same length.
    """
    if len(x_variable) != len(y_variable):
        raise ValueError("The length of x and y must be the same.")
        
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerows(zip(x_variable, y_variable))

def calculate_boltzmann_distribution(energy: float, temperature: float) -> float:
    """
    Calculate the Boltzmann distribution factor for a given energy level and temperature.

    Parameters:
    - energy (float): The energy level in electron volts (eV).
    - temperature (float): The temperature in Kelvin (K). Must be greater than 0.

    Returns:
    - float: The Boltzmann distribution factor for the given energy and temperature.
    """
    
    return 1 / (np.exp(energy / (const.KB * temperature)) - 1)
