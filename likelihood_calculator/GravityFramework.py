import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy import signal

from likelihood_calculator import likelihood_analyser


class GravityFramework:
    def __init__(self):
        self.BDFs = None  # a list of BeadDataFiles
        self.minimizer_1d_results = None  # using x2 only
        self.minimizer_2d_results = None  # using x2 and x3
        self.noise_rms_x2 = 1  # x2 noise gaussian width
        self.noise_rms_x3 = 1  # x3 noise gaussian width
        self.noise_list_x2 = []  # x2 noise values per BDF - sideband
        self.noise_list_x3 = []  # x3 noise values per BDF - sideband
        self.avg_list_x2 = []  # x2 average response - force calibration files
        self.avg_list_x3 = []  # x3 average response - force calibration files
        self.fundamental_freq = 13  # fundamental frequency
        self.Harmonics_list = None  #
        self.Harmonics_array = None  # amplitudes at the given harmonics
        self.Error_array = None  # errors of the amplitudes
        self.scale_X2 = 1  # scale X2 signal to force in Newtons
        self.scale_X3 = 1  # scale X3 signal to force in Newtons

    def plot_dataset(self, bdf_i,  res=50000):
        """
        Plot the x2 and x3 data and their envelopes
        :param res: resolution of the fft
        :param bdf_i: index of BDF to be shown
        """

        bdf = self.BDFs[bdf_i]
        fsamp = 5000
        x2_psd, freqs = matplotlib.mlab.psd(bdf.X2, Fs=fsamp, NFFT=res, detrend='default')
        x3_psd, _ = matplotlib.mlab.psd(bdf.X3, Fs=fsamp, NFFT=res, detrend='default')

        _, ax = plt.subplots(1,2,figsize=(9.5,4))
        ax[0].loglog(freqs, x2_psd)
        ax[1].loglog(freqs, x3_psd)

        plt.show()

