import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from iminuit import Minuit
import time


class LikelihoodAnalyser:
    def __init__(self):
        self.data_x = 0  # x-data to fit
        self.data_y = 0  # y-data to fit
        self.data_y2 = 0  # second y-data to fit
        self.fsamp = 0  # sampling rate
        self.noise_sigma = 1  # gaussian white noise std
        self.noise_sigma2 = 1  # gaussian white noise std

        self.template = None
        self.template2 = None

    def log_likelihood_template(self, alpha, phase, sigma):
        """
        Log likelihood function, using template and control dataset to constrain the noise
        :param alpha: scale factor
        :param phase: phase of the template
        :param sigma: noise
        :return: -2log(likelihood)
        """
        func_t = alpha * np.array(self.template)  # function to minimize
        func_t = np.roll(func_t, int(phase))

        res = sum(np.power(np.abs(self.data_y - func_t), 2))
        print(res, sigma, len(self.data_y), alpha)

        res /= sigma**2
        # res += sum(np.power(np.abs(self.data_y2), 2))/sigma**2
        res += 2*len(self.data_y)*np.log(sigma)
        return res

    def least_squares_template(self, alpha, phase):
        """
        least squares for minimization - any given template
        :param phase: phase of the template
        :param alpha: scale factor
        :return: cost function - sum of squares
        """
        func_t = alpha * np.array(self.template)  # function to minimize
        func_t = np.roll(func_t, int(phase))

        res = sum(np.power(np.abs(self.data_y - func_t), 2))
        return res

    def least_squares_template2(self, alpha, phase):
        """
        least squares for minimization - 2 templates for 2 datasets with shared phase and scale
        :param alpha: scale factor
        :return: cost function - sum of squares
        """
        func_t = alpha * np.array(self.template)  # function to minimize
        func_t = np.roll(func_t, int(phase))

        func_t2 = alpha * np.array(self.template2)  # function to minimize
        func_t2 = np.roll(func_t2, int(phase))

        res = sum(np.power(np.abs(self.data_y - func_t), 2))
        res2 = sum(np.power(np.abs(self.data_y2 - func_t2), 2))

        return res+res2

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
        return res / self.noise_sigma ** 2

    def least_squares_bimodal_sine(self, A, A2, f, phi, phi2):
        """
        least squares for minimization - sine function
        :param phi2: phase second sine
        :param A2: amplitude of second sine
        :param A: Amplitude
        :param f: frequency
        :param phi: phase
        :return: cost function - sum of squares
        """
        func_t = A * np.sin(2 * np.pi * f * self.data_x + phi) + A2 * np.sin(
            2 * np.pi * f * self.data_x + phi2)  # function to minimize
        res = sum(np.power(np.abs(self.data_y - func_t), 2))
        return res / self.noise_sigma ** 2

    def least_squares_2sines(self, A, A2, f, f2, phi, delta_phi):
        """
        least squares for minimization - sine function for two datasets
        :param delta_phi: phase difference between two signals
        :param A, A2: Amplitudes of two signals
        :param f, f2: frequencies of two signals
        :param phi: phase
        :return: cost function - sum of squares
        """
        func_t = A * np.sin(2 * np.pi * f * self.data_x + phi)  # function to minimize
        res = sum(np.power(np.abs(self.data_y - func_t), 2)) / self.noise_sigma ** 2

        func_t2 = A * A2 * np.sin(2 * np.pi * f2 * self.data_x + phi + delta_phi)  # function to minimize
        res2 = sum(np.power(np.abs(self.data_y2 - func_t2), 2)) / self.noise_sigma2 ** 2
        return res + res2

    def least_squares_sine2(self, A, f, phi, sigma):
        """
        least squares for minimization - sine function
        This function takes the white noise width as a parameter as well
        :param sigma: gaussian white noise variance
        :param A: Amplitude
        :param f: frequency
        :param phi: phase
        :return: cost function - sum of squares
        """
        func_t = A * np.sin(2 * np.pi * f * self.data_x + phi)  # function to minimize
        res = sum(np.power(np.abs(self.data_y - func_t), 2))
        print(res, sigma, len(self.data_x), A)

        res /= sigma**2
        res += 2*len(self.data_x) * np.log(sigma)

        return res

    def find_mle_template(self, x, template, center_freq, bandwidth, **kwargs):
        """
        The function is fitting the data with a  template using iminuit.
        The fitting is done after applying a bandpass filter to both the template and the data.
        :param template: template for the fit
        :param bandwidth: bandwidth for butter filter [Hz]
        :param center_freq: center frequency for the bandpass filter
        :param x: 1 dim. position data (time domain)
        :return: minimizer result
        """
        # filtering the template and the data
        b, a = signal.butter(3, [2. * (center_freq - bandwidth / 2.) / self.fsamp,
                                 2. * (center_freq + bandwidth / 2.) / self.fsamp], btype='bandpass')
        self.data_y = signal.filtfilt(b, a, x)[5000:-5000]
        self.template = signal.filtfilt(b, a, template)[5000:-5000]

        mimuit_minimizer = Minuit(self.least_squares_template, **kwargs)
        mimuit_minimizer.migrad(ncall=50000)
        return mimuit_minimizer

    def find_mle_PL(self, x, template, scale, center_freq, noise_freq, bandwidth, decimate=10, **kwargs):
        """
        The function is fitting the data with a template using iminuit and the likelihood function
        The fitting is done after applying a bandpass filter to both the template and the data.
        :param decimate: decimate data (good for correlated datasets)
        :param template: template for the fit
        :param bandwidth: bandwidth for butter filter [Hz]
        :param center_freq: center frequency for the bandpass filter
        :param noise_freq: noise bandwidth
        :param x: 1 dim. position data (time domain)
        :return: minimizer result
        """
        # filtering the template and the data
        # b, a = signal.butter(3, [2. * (center_freq - bandwidth / 2.) / self.fsamp,
        #                          2. * (center_freq + bandwidth / 2.) / self.fsamp], btype='bandpass')
        b, a = signal.butter(3, [2. * (20 - bandwidth / 2.) / self.fsamp,
                                 2. * (20 + bandwidth / 2.) / self.fsamp], btype='bandpass')
        self.data_y = signal.filtfilt(b, a, x)[5000:-5000:decimate]
        print(np.std(self.data_y))
        tmp_y = signal.filtfilt(b, a, x)[5000:-5000]
        print(np.std(x), np.std(tmp_y), np.std(tmp_y[::decimate]))
        self.template = signal.filtfilt(b, a, template)[5000:-5000:decimate]*scale

        b, a = signal.butter(3, [2. * (noise_freq - bandwidth / 2.) / self.fsamp,
                                 2. * (noise_freq + bandwidth / 2.) / self.fsamp], btype='bandpass')
        self.data_y2 = signal.filtfilt(b, a, x)[5000:-5000:decimate]  # x3 data - QPD carrier phase

        mimuit_minimizer = Minuit(self.log_likelihood_template, **kwargs)
        mimuit_minimizer.migrad(ncall=50000)
        return mimuit_minimizer

    def find_mle_template2(self, x2, template2, x3, template3, center_freq, bandwidth, decimate, **kwargs):
        """
        The function is fitting the data with a  template using iminuit.
        The fitting is done after applying a bandpass filter to both the template and the data.
        :param template: template for the fit
        :param bandwidth: bandwidth for butter filter [Hz]
        :param center_freq: center frequency for the bandpass filter
        :param x: 1 dim. position data (time domain)
        :return: minimizer result
        """
        # filtering the template and the data
        b, a = signal.butter(3, [2. * (center_freq - bandwidth / 2.) / self.fsamp,
                                 2. * (center_freq + bandwidth / 2.) / self.fsamp], btype='bandpass')
        self.data_y = signal.filtfilt(b, a, x2)[5000:-5000:decimate]  # x2 data - QPD carrier amplitude
        self.template = signal.filtfilt(b, a, template2)[5000:-5000:decimate]  # x2 template

        self.data_y2 = signal.filtfilt(b, a, x3)[5000:-5000:decimate]  # x3 data - QPD carrier phase
        self.template2 = signal.filtfilt(b, a, template3)[5000:-5000:decimate]  # x3 template

        mimuit_minimizer = Minuit(self.least_squares_template2, **kwargs)
        mimuit_minimizer.migrad(ncall=50000)

        return mimuit_minimizer

    def find_mle_sin(self, x, drive_freq=0, fsamp=5000, bandwidth=50, noise_rms=0, plot=False, suppress_print=True,
                     bimodal=False, **kwargs):
        """
        The function is fitting the data with a sine template using iminuit.
        The fitting is done after applying a bandpass filter.
        :param bimodal: bimodal sine function
        :param suppress_print: suppress all printing
        :param noise_rms: std of the white gaussian noise
        :param plot: plot the data and its fft
        :param bandwidth: bandwidth for butter filter [Hz]
        :param fsamp: sampling rate [1/sec]
        :param x: 1 dim. position data (time domain)
        :param drive_freq: drive frequency of the response
        :return: estimated values, chi_square
        """
        if not suppress_print:
            print('Data overall time: ', len(x) / fsamp, ' sec.')
        self.fsamp = fsamp
        if noise_rms != 0:
            self.noise_sigma = noise_rms

        # apply a bandpass filter to data and store data in the correct place for the minimization
        self.data_x = np.arange(0, len(x)) / fsamp
        start = time.time()
        if drive_freq != 0:
            if not suppress_print:
                print('Bandpass filter ON. Bandwidth: ', bandwidth, 'Hz')
            b, a = signal.butter(3, [2. * (drive_freq - bandwidth / 2.) / self.fsamp,
                                     2. * (drive_freq + bandwidth / 2.) / self.fsamp], btype='bandpass')
            self.data_y = signal.filtfilt(b, a, x)
        else:
            self.data_y = x
        end = time.time()
        if not suppress_print:
            print('bandpass time: ', end - start)
        print(np.std(self.data_y))
        # we create an instance of Minuit and pass the function to minimize
        if bimodal:
            mimuit_minimizer = Minuit(self.least_squares_bimodal_sine, **kwargs)
        else:
            if 'sigma' in kwargs.keys():
                mimuit_minimizer = Minuit(self.least_squares_sine2, **kwargs)
            else:
                mimuit_minimizer = Minuit(self.least_squares_sine, **kwargs)

        start = time.time()
        mimuit_minimizer.migrad(ncall=50000)
        end = time.time()
        if not suppress_print:
            print('minimization time: ', end - start)
            print(mimuit_minimizer.get_param_states())

        if plot:
            _, ax = plt.subplots(1, 2, figsize=(9.5, 4))
            ax[0].scatter(self.data_x, self.data_y)
            fft = np.abs(np.fft.rfft(x)) ** 2
            freq = np.fft.rfftfreq(len(x), d=1. / fsamp)
            ax[0].set(title='raw data')
            plt.subplot(122)
            mimuit_minimizer.draw_profile('A', subtract_min=True)
            plt.show()
        if not suppress_print:
            print('reduced chi2: ', mimuit_minimizer.fval / (len(self.data_y) - 2))

        return mimuit_minimizer

    def find_mle_2sin(self, x, x2, drive_freq=0, fsamp=5000, bandwidth=50, noise_rms=0, noise_rms2=0, plot=False,
                      suppress_print=True,
                      **kwargs):
        """
        The function is fitting the data with a sine template using iminuit.
        The fitting is done after applying a bandpass filter.
        :param suppress_print: suppress all printing
        :param noise_rms, noise_rms2: std of the white gaussian noise
        :param plot: plot the data and its fft
        :param bandwidth: bandwidth for butter filter [Hz]
        :param fsamp: sampling rate [1/sec]
        :param x, x2: two 1-dim. position datasets (time domain)
        :param drive_freq: drive frequency of the response
        :return: estimated values, chi_square
        """
        if not suppress_print:
            print('Data overall time: ', len(x) / fsamp, ' sec.')
        self.fsamp = fsamp
        if noise_rms != 0:
            self.noise_sigma = noise_rms
            self.noise_sigma2 = noise_rms2

        # apply a bandpass filter to data and store data in the correct place for the minimization
        self.data_x = np.arange(0, len(x)) / fsamp
        start = time.time()
        if drive_freq != 0:
            if not suppress_print:
                print('Bandpass filter ON. Bandwidth: ', bandwidth, 'Hz')
            b, a = signal.butter(3, [2. * (drive_freq - bandwidth / 2.) / self.fsamp,
                                     2. * (drive_freq + bandwidth / 2.) / self.fsamp], btype='bandpass')
            self.data_y = signal.filtfilt(b, a, x)
            self.data_y2 = signal.filtfilt(b, a, x2)
        else:
            self.data_y = x
            self.data_y2 = x2
        end = time.time()
        if not suppress_print:
            print('bandpass time: ', end - start)

        # we create an instance of Minuit and pass the function to minimize
        mimuit_minimizer = Minuit(self.least_squares_2sines, **kwargs)

        start = time.time()
        mimuit_minimizer.migrad(ncall=50000)
        end = time.time()
        if not suppress_print:
            print('minimization time: ', end - start)
            print(mimuit_minimizer.get_param_states())

        if plot:
            _, ax = plt.subplots(1, 2, figsize=(9.5, 4))
            # ax[0].scatter(self.data_x, self.data_y)
            # fft = np.abs(np.fft.rfft(x)) ** 2
            # freq = np.fft.rfftfreq(len(x), d=1. / fsamp)
            # ax[0].set(title='raw data')
            plt.subplot(121)
            mimuit_minimizer.draw_profile('A', subtract_min=True)
            plt.subplot(122)
            mimuit_minimizer.draw_profile('A2', subtract_min=True)
            plt.show()
        if not suppress_print:
            print('reduced chi2: ', mimuit_minimizer.fval / (2 * len(self.data_y) - 4))

        return mimuit_minimizer
