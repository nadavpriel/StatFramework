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
        self.fsamp = 5000
        self.lc_i = likelihood_analyser.LikelihoodAnalyser()

    def plot_dataset(self, bdf_i, res=50000):
        """
        Plot the x2 and x3 data and their envelopes
        :param res: resolution of the fft
        :param bdf_i: index of BDF to be shown
        """

        bdf = self.BDFs[bdf_i]
        x2_psd, freqs = matplotlib.mlab.psd(bdf.x2 * 50000, Fs=self.fsamp, NFFT=res, detrend='default')
        x3_psd, _ = matplotlib.mlab.psd(bdf.x3 / 6, Fs=self.fsamp, NFFT=res, detrend='default')

        _, ax = plt.subplots(1, 2, figsize=(9.5, 4))
        ax[0].loglog(freqs, x2_psd)
        ax[1].loglog(freqs, x3_psd)

        plt.show()

    def get_amplitude(self, bdf_i, harmonic_num, noise_rms, noise_rms2, bandwidth=1, **fit_kwargs):
        """
        Fit and extract the amplitude of one harmonic from one particular file
        :param harmonic_num: which harmonic to fit
        :param bandwidth: bandpass bandwidth
        :param noise_rms, noise_rms2: noise std of X2 and X3
        :param bdf_i: index of the bdf dataset to be used
        :return: amplitude, error
        """
        bb = self.BDFs[bdf_i]
        frequency = self.fundamental_freq * harmonic_num

        xx2 = bb.response_at_freq2('x', frequency, bandwidth=bandwidth) * 50000
        xx2 = xx2[5000:-5000]  # cut out the first and last second

        xx3 = bb.response_at_freq3('x', frequency, bandwidth=bandwidth) / 6
        xx3 = xx3[5000:-5000]  # cut out the first and last second

        m1_tmp = self.lc_i.find_mle_2sin(xx2, xx3, fsamp=self.fsamp,
                                         noise_rms=noise_rms,
                                         noise_rms2=noise_rms2,
                                         plot=False, suppress_print=True, **fit_kwargs)

        print('***************************************************')
        print('bdf_i: ', bdf_i, ', frequency: ', frequency)
        print('X2-amplitude: ', '{:.2e}'.format(np.abs(m1_tmp.values[0])))
        print('reduced chi2: ', m1_tmp.fval / (len(xx2) - 3))

        return m1_tmp.values[0], m1_tmp.errors[0]

    def build_noise_array(self, sideband_freq, bandwidth=1):

        self.noise_list_x2 = []
        self.noise_list_x3 = []

        for bb in self.BDFs:
            xx2 = bb.response_at_freq2('x', sideband_freq, bandwidth=bandwidth) * 50000
            self.noise_list_x2.append(np.std(xx2[5000:-5000]))

            xx3 = bb.response_at_freq3('x', sideband_freq, bandwidth=bandwidth) / 6
            self.noise_list_x3.append(np.std(xx3[5000:-5000]))

        self.noise_rms_x2 = np.mean(self.noise_list_x2)
        self.noise_rms_x3 = np.mean(self.noise_list_x3)
        print('x2 noise rms: ', self.noise_rms_x2)
        print('x3 noise rms: ', self.noise_rms_x3)