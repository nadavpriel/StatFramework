import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy import signal

from likelihood_calculator import likelihood_analyser


class DMAnalyser:
    def __init__(self):
        self.BDFs = None  # a list of BeadDataFiles
        self.minimizer_1d_results = None  # using x2 only
        self.minimizer_2d_results = None  # using x2 and x3
        self.noise_rms_x2 = 1  # x2 noise gaussian width
        self.noise_rms_x3 = 1  # x3 noise gaussian width
        self.noise_list_x2 = []  # x2 noise values per  BDF
        self.noise_list_x3 = []  # x3 noise values per BDF
        self.avg_list_x2 = []  # x2 average response
        self.avg_list_x3 = []  # x3 average response

    def estimate_noise(self, bandwidth=10, frequency=151):
        """
        Estimation of the noise level outside of the signal-band
        :param frequency: The frequency of the carrier signal
        :type bandwidth: bandpass filter bandpass
        """
        for i, bb in enumerate(self.BDFs):
            xx3 = bb.response_at_freq3('x', frequency+bandwidth, bandwidth=bandwidth) / 6
            analytic_signal3 = signal.hilbert(xx3)
            amplitude_envelope3 = np.abs(analytic_signal3)
            self.noise_list_x3.appebnd(np.std(amplitude_envelope3[5000:-5000]))

            xx2 = bb.response_at_freq2('x', frequency+bandwidth, bandwidth=bandwidth) / 50000
            analytic_signal2 = signal.hilbert(xx2)
            amplitude_envelope2 = np.abs(analytic_signal2)
            self.noise_list_x2.appebnd(np.std(amplitude_envelope2[5000:-5000]))
        self.noise_rms_x2 = np.mean(self.noise_list_x2)
        self.noise_rms_x3 = np.mean(self.noise_list_x3)
        print('x2 noise rms: ', self.noise_rms_x2)
        print('x3 noise rms: ', self.noise_rms_x3)
