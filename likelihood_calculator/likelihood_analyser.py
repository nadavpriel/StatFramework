import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from iminuit import Minuit
import pprint


class LikelihoodAnalyser:
    def __init__(self):
        print('Constructor')
        self.data_x = 0  # x-data to fit
        self.data_y = 0  # y-data to fit
        self.fsamp = 0  # sampling rate

    def least_squares_sine(self, A, f, phi):
        """
        least squares for minimization - sine function
        :param A: Amplitude
        :param f: frequency
        :param phi: phase
        :return: cost function - sum of squares
        """
        func_t = A * np.sin(2 * np.pi * f * self.data_x + phi)  # function to minimize
        res = sum(np.power(np.abs(self.data_y - func_t), 2))
        return res / 100

    def find_mle_sin(self, x, drive_freq=0, fsamp=5000, bandwidth=50, plot=False):
        """
        The function is fitting the data with a sine template using iminuit.
        The fitting is done after applying a bandpass filter.
        :param plot: plot the data and its fft
        :param bandwidth: bandwidth for butter filter [Hz]
        :param fsamp: sampling rate [1/sec]
        :param x: 1 dim. position data (time domain)
        :param drive_freq: drive frequency of the response
        :return: estimated values, chi_square
        """
        print('Starting to fit a sine template')
        print('data overall time: ', len(x) / fsamp, ' sec.')
        self.fsamp = fsamp

        # apply bandpass filter to data and store data in the correct place for the minimization
        self.data_x = np.arange(0, len(x)) / fsamp
        if drive_freq != 0:
            print('Bandpass filter on. Bandwidth: ', bandwidth)
            b, a = signal.butter(3, [2. * (drive_freq - bandwidth / 2.) / self.fsamp,
                                     2. * (drive_freq + bandwidth / 2.) / self.fsamp], btype='bandpass')
            self.data_y = signal.filtfilt(b, a, x)
        else:
            self.data_y = x

        # fit arguments
        fit_kwargs = {'A': 5, 'f': 100, 'phi': 0, 'error_A': 2, 'error_f': 10, 'error_phi': 0.5, 'errordef': 0.5,
                      'limit_phi': [0, 2 * np.pi],
                      'print_level': 0, 'fix_f': True, 'fix_phi': False}

        # we create an instance of Minuit and pass the function to minimize
        m = Minuit(self.least_squares_sine, **fit_kwargs)
        m.migrad(ncall=50000)
        print(m.get_param_states())

        if plot:
            _, ax = plt.subplots(1, 2, figsize=(9.5, 4))
            ax[0].scatter(self.data_x, self.data_y)
            fft = np.abs(np.fft.rfft(x)) ** 2
            freq = np.fft.rfftfreq(len(x), d=1. / fsamp)
            ax[1].semilogy(freq, fft)
            ax[1].set(xlim=(1, 500))
            plt.show()

        print('reduced chi2: ', m.fval / (len(self.data_y) - 2))

        m.draw_profile('A', subtract_min=True)
        plt.show()

        return m
