import numpy as np
from tqdm import tqdm
from pycbc.waveform import get_fd_waveform
import pycbc.psd
from pycbc.psd import aLIGOZeroDetHighPower
import h5py

# Read the bank 
with h5py.File('template_bank_2_300_30Hz.h5', 'r') as bank_file:
    mass1 = bank_file['mass1'][:]

hp_fd_waves = []
freqs = None  # Will store the sample frequencies (same for all)

# Create the fd waveform
for m1 in tqdm(mass1):
    # Generate a template to filter with
    hp, hc = get_fd_waveform(approximant="SEOBNRv4_ROM", mass1=m1,
                              mass2=m1, f_lower=30, delta_f=1.0/80)
    hp.resize(163840 // 2 + 1)
    #print(hp)
    #print(hp.sample_frequencies)

    # Convert the list to a NumPy array
    hp_fd_waves.append(hp.numpy())
    if freqs is None:
        freqs = hp.sample_frequencies.numpy()

hp_fd_waves = np.array(hp_fd_waves, dtype=np.complex64)
freqs = np.array(freqs, dtype=np.float32) 
    
with h5py.File("template_bank_2_300_30Hz_fd.h5", "w") as f:
    f.create_dataset("strain", data=hp_fd_waves, dtype=np.complex128)  # Save complex 
    f.create_dataset("frequencies", data=freqs, dtype=np.float64)  # Save frequencies
       
