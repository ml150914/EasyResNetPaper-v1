#!/home/lorenzo-mobilia/.conda/envs/myenv3/bin/python 

import matplotlib.pyplot as plt
from pycbc.noise import noise_from_psd
import pycbc.psd
from pycbc.psd import aLIGOZeroDetHighPower
import pylab
import random
import pycbc.frame as frame
import numpy as np
from pycbc.waveform import get_td_waveform, get_fd_waveform, get_td_waveform_from_fd
from pycbc.waveform.sinegauss import fd_sine_gaussian
from tqdm import tqdm
import os
from pycbc.psd import welch, interpolate
from pycbc.filter import resample_to_delta_t
from pycbc.filter import highpass_fir, matched_filter, sigma
import sys
from tqdm import tqdm
import argparse
"""
### This macro will produce a bns signal injected  at 61s of 80s long gaussian noise.
### The production parameters such as masses and distance are randomized.
### The masses production is uniform, while the distance is d^2.
### The injection is perfomed in an interval of 4s at random time at 61s.
### The files are saved in files named Injections_*.gwf with a unique index 
### The parameters of each injections (masses, distance, ...) are saved in injection_param_*.txt
"""
"""
from pycbc.waveform import td_approximants, fd_approximants

# Print the list of available time-domain waveform approximants
print("Available TD Approximants in PyCBC:")
for approx in fd_approximants():
    print(approx)
"""
parser = argparse.ArgumentParser(usage='',
    description="Generate the Time-Template map")
parser.add_argument("--analysis", type=str, required=True,
                    help="Analysis parameter")
parser.add_argument("--job-id", type=str, required=True,
                    help="Job identifier")
parser.add_argument("--type-inj", type=str, required=True,
                    help='type of injection (mixed, bns, bbh)')
args = parser.parse_args()

def safe_generate(m1, m2, spin1x, spin1y, spin1z, spin2x, spin2y, spin2z,
                   incl, distance, delta_t=1/32768, f_lower=27, max_tries=20):
    for attempt in range(max_tries):
        try:
            hp, hc = get_td_waveform(approximant="SEOBNRv4PHM",
                                      mass1=m1, mass2=m2,
                                      spin1x=spin1x, spin1y=spin1y, spin1z=spin1z,
                                      spin2x=spin2x, spin2y=spin2y, spin2z=spin2z,
                                      inclination=incl,
                                      delta_t=delta_t,
                                      mode_array=[(2, 2)],
                                      f_lower=f_lower,
                                      distance=distance)
            return hp, hc
        except RuntimeError as e:
            # Any RuntimeError here means SEOBNRv4PHM rejected this
            # parameter point (Nyquist, ringdown attachment, extremal
            # spin, etc.) — redraw masses/spins and try again.
            print(f"[safe_generate] attempt {attempt}: waveform failed "
                  f"(m1={m1:.3f}, m2={m2:.3f}) -> {e}. Redrawing.")
            m1 = random.uniform(1.4, 3)
            m2 = random.uniform(1.4, 3)
            mc = chirp_mass(m1, m2)
            d = generate_x2_distribution(x_min_bns, x_max_bns)
            distance = chirp_distance(mc, d)
        except Excpetion:
            raise
    raise RuntimeError("Could not generate waveform after max_tries")


# The color of the noise matches a PSD which you provide
f_low = 20.0
delta_f = 1.0 / 128
delta_t = 1.0 / 2048
flen = int(128 * 2048) + 1 # you have to correct with tlen / 2 + 1
psd = pycbc.psd.aLIGOZeroDetHighPower(flen, delta_f, f_low)

lines = []

# Extremal points for the distance
# change those accordingly to have louder or quiter signals
x_min_bns = 50
x_max_bns = 538

x_min_bbh = 50
x_max_bbh = 538

analysis = args.analysis
job_id = int(args.job_id)
type_inj = args.type_inj
# Read the argument to generate noise pr injection 
if(analysis == 'injection'):
    path_folders = '.'
    
elif(analysis == 'noise'):
    path_folders = ''


num_injection = 1
for i in tqdm(range(num_injection)):
    # -------> Generate the injections parameters
    job_id_save = 1
    seed = job_id_save + 663716831
    populator = random.uniform(0,1)
    if(type_inj == 'mixed'):
        if populator < 1.1:
            m1 = 1.4
            m2 = 25
            mc = pycbc.conversions.mchirp_from_mass1_mass2(m1, m2)
            spin1z = 0
            spin2z = 0
            spin1x = 0.5
            spin2x = 0.7
            spin1y = 0.5
            spin2y = 0.7
            incl = 1.54
            d = 50
            distance = d
            
    # ------> Generate the injections
    #hp, hc = get_td_waveform(approximant="SEOBNRv4PHM",
    #                         mass1=m1,
    #                         mass2=m2,
    #                         spin1z = spin1z,
    #                         spin2z = spin2z,
    #                         spin1x = spin1x,
    #                         spin2x = spin2x,
    #                         spin1y = spin1y,
    #                         spin2y = spin2y,
    #                         inclination = incl,
    #                         delta_t=1 / 32768,
    #                         mode_array = [(2,2)],
    #                         f_lower=27,
    #                         distance = distance)
    hp, hc = safe_generate(m1, m2, spin1x, spin1y, spin1z, spin2x, spin2y, spin2z,
                           incl, distance, delta_t=1/32768, f_lower=27, max_tries=300)
    # Let the signal begin in  the 61.3 -61.5  s window
    hp.start_time = 61 - hp.duration + random.uniform(0.3,0.5)
    merge_time = hp.end_time

    # ------> Generate the noise
    tsamples = int(80 * 2048) # generate 80 seconds of noise
    noise = pycbc.noise.noise_from_psd(tsamples, delta_t, psd, seed = seed)
    hp = resample_to_delta_t(hp, 1.0/2048)  # resample the signal to avoid over-computation
    noise_zero = noise * 0 # create a zero time series

    if(analysis == 'injection'):
        injection_ts = noise_zero.inject(hp) # inject the signal into this zero ts
    #injection_ts = injection_ts.inject(sineGaussian_ts)
    elif analysis == 'noise':
        injection_ts = noise_zero
    #injection_ts = injection_ts.inject(sineGaussian_ts)

    # -------> Inject the signal in noise
    data = noise + injection_ts # inject the bns

    # ------> Calculate the optimal snr
    # Calculate the optimal snr
    psd_est = interpolate(welch(noise), 1.0 / injection_ts.duration)
    optimal_snr = pycbc.filter.sigma(injection_ts, psd=psd_est,
                                     low_frequency_cutoff=40)

    # Generate a random number, that is the activator
    # here we impose that only 30% of datastrain will be glitched-affected
    activator = random.uniform(0, 1)
    threshold = 1.1
    # -------> inject here the glitch If above threshold)
    if activator >= threshold:
        data = data + sineGaussian_ts # inject the sinegaussian glitch on 20% of data
    if(analysis == 'injection'):
        name_file = f'/injection_{job_id_save}.gwf' # save the injection file                       
        frame.write_frame(path_folders  + name_file, 'L1', data)
        injections_params = {"job_id": job_id_save,
                             "activator" : activator,
                             "distance": distance,
                             "m1": m1,
                             "m2": m2,
                             "s1z": spin1z,
                             "s2z": spin2z,
                             "s1x": spin1x,
                             "s2x": spin2x,
                             "s1y": spin1y,
                             "s2y": spin2y,
                             "merge_time": merge_time,
                             "optimal_snr": optimal_snr,
                             "seed": seed,
                             "populator": populator,
                             "detector": 'L1'}

    elif(analysis == 'noise'):
        name_file = f'/noise_{job_id_save}.gwf'
        frame.write_frame(path_folders  + name_file, 'L1', data)

        # --------> plt the injection to have a visual representation
        #plot(injection_ts, noise, sineGaussian_ts, sineGaussian_td, analysis, threshold, activator, type_inj, job_id)
        # Save the block of injections
    if(analysis == 'injection'):
        with open(path_folders + f'/injection_param_{job_id_save}.txt', 'w') as f:
            for key, value in injections_params.items():
                f.write(f"{key}: {value}\n")

    elif(analysis == 'noise'):
        with open(path_folders + f'/sineGauss_param_{job_id}.txt', 'w') as f:
            sineGaussian_params = {"job_id": job_id,
                                   "activator" : activator,
                                   "seed": seed,
                                   "f_0" : f_0,
                                   "q_factor" : q_factor,
                                   "amplitude" : amplitude,
                                   "phase" : phase,
                                   "time_jitter" : time_jitter,
                                   "detector": 'L1'}
            for key, value in sineGaussian_params.items():
                f.write(f"{key}: {value}\n")
