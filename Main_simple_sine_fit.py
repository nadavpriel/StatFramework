import numpy as np
from likelihood_calculator import likelihood_analyser

print('Starting main')
lc_i = likelihood_analyser.LikelihoodAnalyser()

# generating fake data
samples = 50000
noise_rms = 10
time = np.arange(0, samples)/5000
freq = 100
amp = 1
sig_x = amp * np.sin(2 * np.pi * freq * time) + noise_rms * np.random.randn(samples)

x = np.random.randn(samples)
lc_i.noise_sigma = noise_rms
m = lc_i.find_mle_sin(sig_x, drive_freq=0, bandwidth=70)
