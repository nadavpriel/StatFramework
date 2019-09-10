import numpy as np

from likelihood_calculator import likelihood_analyser
import time

import LeastSquares

print('************************')
print('Starting simple sine fit')
print('************************')

lc_i = likelihood_analyser.LikelihoodAnalyser()

# generating fake data
samples = 500000
noise_rms = 1
time_x = np.arange(0, samples) / 5000
freq = 100
amp = 1
sig_x = amp * np.sin(2 * np.pi * freq * time_x) + noise_rms * np.random.randn(samples)

x = np.random.randn(samples)

# fit arguments
fit_kwargs = {'A': 5, 'f': 100, 'phi': 0, 'error_A': 2, 'error_f': 10, 'error_phi': 0.5, 'errordef': 0.5,
              'limit_phi': [0, 2 * np.pi],
              'print_level': 0, 'fix_f': True, 'fix_phi': False}
m = lc_i.find_mle_sin(sig_x, drive_freq=freq, noise_rms=noise_rms, bandwidth=100, plot=False, **fit_kwargs)

m = lc_i.find_mle_sin2(sig_x, drive_freq=freq, noise_rms=noise_rms, bandwidth=100, plot=False, **fit_kwargs)
