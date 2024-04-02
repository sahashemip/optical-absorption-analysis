import matplotlib.pyplot as plt

class ResultData:
    def __init__(self):
        self.__optical_absorption_spectrum = None
        self.__photon_e = None

    @property
    def optical_absorption_spectrum(self):
        return self.__optical_absorption_spectrum

    @optical_absorption_spectrum.setter
    def optical_absorption_spectrum(self, value):
        self.__optical_absorption_spectrum = value

    @property
    def photon_e(self):
        return self.__photon_e

    @photon_e.setter
    def photon_e(self, value):
        self.__photon_e = value

def save_plot(zpl_energy, algorithm_result: ResultData, show=True, write=True):
    plt.plot(zpl_energy - algorithm_result.photon_e, algorithm_result.optical_absorption_spectrum, linestyle='--',
             color='blue')
    pl_x = zpl_energy - algorithm_result.photon_e
    pl_y = algorithm_result.optical_absorption_spectrum

    if write:
        write_to_file('pl_spectrum_I.dat', pl_x, pl_y)

    if show:
        plt.show()