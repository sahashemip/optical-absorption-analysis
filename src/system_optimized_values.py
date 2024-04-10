import numpy as np
import constants as cnts


class SystemOptimizedValues:
    '''
    A class for optimization, and reconstruction of
    key quantities that are used throughout formulations.
    
    Attributes:
    
    Methods:
    
    '''
    def __init__(self, smvi, masses, zpl_energy):
        self.smvi = smvi
        self.masses = masses
        self.zpl_energy = zpl_energy       

    @property
    def diag_masses(self):
        mass_tmp = np.repeat(self.masses, 3)
        mass = np.diag(np.sqrt(mass_tmp))
        return mass

    @property
    def phE(self):
        phE = self.zpl_energy
        return phE

    @property
    def ngrids(self):
        ngrids = int(self.phE * 1e3)
        return ngrids

    @property
    def T(self):
        return 3.5

    @property
    def tgrids(self):
        tgrids = 2 * self.ngrids
        return tgrids

    @property
    def dE(self):
        return self.phE / self.ngrids

    @property
    def energy(self):
        return np.linspace(0.001, self.phE, self.ngrids)

    @property
    def total_time(self):
        return np.linspace(-self.T, self.T, self.tgrids)

    @property
    def phonE_eV(self):
        return cnts.HBAR_TO_EVS * cnts.TERA * 2 * np.pi * np.array(self.smvi.phonon_energy)

    @property
    def dq2_units_convers(self):
        return (cnts.ANGSTROM ** 2) * cnts.AMU_TO_KG * cnts.JOULE_TO_EV

    @property
    def photon_energy(self):
        return  cnts.HBAR_TO_EVS * (2 * np.pi / self.T / cnts.PICO) * np.arange(self.tgrids)
