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
import os
from pycbc.psd import welch, interpolate
from pycbc.filter import resample_to_delta_t
from pycbc.filter import highpass_fir, matched_filter, sigma
import sys
from sineGauss import sineGaussian
from plot_TS import plot
from distributions_Mc_distance import generate_x2_distribution, bbh_distribution_law, chirp_mass, chirp_distance
import argparse
import configparser
import time
from tqdm import tqdm
import json 
"""
### This macro will produce a bns signal injected  at 61s of 80s long gaussian noise.
### The production parameters such as masses and distance are randomized.
### The masses production is uniform, while the distance is d^2.
### The injection is perfomed in an interval of 4s at random time at 61s.
### The files are saved in files named Injections_*.gwf with a unique index 
### The parameters of each injections (masses, distance, ...) are saved in injection_param_*.txt
"""

parser = argparse.ArgumentParser(usage='', description="Generate the Time-Template map")

parser.add_argument("--config",                          type=str,   default=None,  help="Path to config file")
parser.add_argument("--analysis",                          type=str,   default=None,  help="Analysis parameter")
parser.add_argument("--job-id",                          type=str,   required=True, help="Job identifier")
parser.add_argument("--injection-type",                  type=str,   default=None,  help="type of injection (mixed, bns, bbh)")
parser.add_argument("--number-injections",               type=int,   default=None,   help="number of injections per batch") 
parser.add_argument("--min-distance-bns",                type=float, default=None,  help="Minimum distance for bns population (power-law)")
parser.add_argument("--max-distance-bns",                type=float, default=None,  help="Maximum distance for bns population (power-law)")
parser.add_argument("--min-distance-bbh",                type=float, default=None,  help="Minimum distance for bbh population (power-law)")
parser.add_argument("--max-distance-bbh",                type=float, default=None,  help="Maximum distance for bbh population (power-law)")
parser.add_argument("--seed",                            type=int,   default=None,  help="Int value for seed")
parser.add_argument("--population-fraction",             type=float, default=None,  help="Fraction of bns and bbh population")
parser.add_argument("--min-m-bns",                       type=float, default=None,  help="Minimum mass value for bns population")
parser.add_argument("--max-m-bns",                       type=float, default=None,  help="Maximum mass value for bns population")
parser.add_argument("--min-m-bbh",                       type=float, default=None,  help="Minimum mass value for bbh population")
parser.add_argument("--max-m-bbh",                       type=float, default=None,  help="Maximum mass value for bbh population")
parser.add_argument("--min-sz-bns",                      type=float, default=None,  help="Minimum spin-z value for bns population")
parser.add_argument("--max-sz-bns",                      type=float, default=None,  help="Maximum spin-z value for bns population")
parser.add_argument("--min-sz-bbh",                      type=float, default=None,  help="Minimum spin-z value for bbh population")
parser.add_argument("--max-sz-bbh",                      type=float, default=None,  help="Maximum spin-z value for bbh population")
parser.add_argument("--low-frequency-generating-injections", type=int, default=None, help="Minimum frequency to inject the signal")
parser.add_argument("--glitch-threshold",                type=float, default=None,  help="Threshold for polluting data")
parser.add_argument("--path-saving-data",                type=str,   default=None,  help="Path to save the data")

args = parser.parse_args()

# Load config file and fill in any unset args
if args.config:
    config = configparser.ConfigParser()
    config.read(args.config)
    cfg = config["general"]

    def get_float(key): 
        val = cfg.get(key)
        return None if (val is None or val.strip().lower() == 'none') else float(val)
    def get_int(key):   
        val = cfg.get(key)
        return None if (val is None or val.strip().lower() == 'none') else int(val)
    def get_str(key):   
        val = cfg.get(key)
        return None if (val is None or val.strip().lower() == 'none') else val

    if args.analysis                        is None: args.analysis                        = get_str  ("analysis")
    if args.injection_type                  is None: args.injection_type                  = get_str  ("injection_type")
    if args.number_injections               is None: args.number_injections               = get_int  ("number_injections")
    if args.min_distance_bns                is None: args.min_distance_bns                = get_float("min_distance_bns")
    if args.max_distance_bns                is None: args.max_distance_bns                = get_float("max_distance_bns")
    if args.min_distance_bbh                is None: args.min_distance_bbh                = get_float("min_distance_bbh")
    if args.max_distance_bbh                is None: args.max_distance_bbh                = get_float("max_distance_bbh")
    if args.seed                            is None: args.seed                            = get_int  ("seed")
    if args.population_fraction             is None: args.population_fraction             = get_float("population_fraction")
    if args.min_m_bns                       is None: args.min_m_bns                       = get_float("min_m_bns")
    if args.max_m_bns                       is None: args.max_m_bns                       = get_float("max_m_bns")
    if args.min_m_bbh                       is None: args.min_m_bbh                       = get_float("min_m_bbh")
    if args.max_m_bbh                       is None: args.max_m_bbh                       = get_float("max_m_bbh")
    if args.min_sz_bns                      is None: args.min_sz_bns                      = get_float("min_sz_bns")
    if args.max_sz_bns                      is None: args.max_sz_bns                      = get_float("max_sz_bns")
    if args.min_sz_bbh                      is None: args.min_sz_bbh                      = get_float("min_sz_bbh")
    if args.max_sz_bbh                      is None: args.max_sz_bbh                      = get_float("max_sz_bbh")
    if args.low_frequency_generating_injections is None: args.low_frequency_generating_injections = get_int("low_frequency_generating_injections")
    if args.glitch_threshold                is None: args.glitch_threshold                = get_float("glitch_threshold")
    if args.path_saving_data                is None: args.path_saving_data                = get_str  ("path_saving_data")

# Set defaults for optional args if still unset
if args.population_fraction             is None: args.population_fraction             = 0.5
if args.low_frequency_generating_injections is None: args.low_frequency_generating_injections = 27

# Validate all required args are now set
required_args = [
    "analysis",
    "seed",
    "glitch_threshold",
    "path_saving_data"
]
for arg in required_args:
    if getattr(args, arg) is None:
        parser.error(f"--{arg.replace('_', '-')} is required (via CLI or config file)")
start = time.perf_counter()
# The color of the noise matches a PSD which you provide
f_low = 20.0
delta_f = 1.0 / 128
delta_t = 1.0 / 2048
flen = int(128 * 2048) + 1 # you have to correct with tlen / 2 + 1
psd = pycbc.psd.aLIGOZeroDetHighPower(flen, delta_f, f_low)

# Extremal points for the distance
# change those accordingly to have louder or quiter signals
x_min_bns = args.min_distance_bns
x_max_bns = args.max_distance_bns

x_min_bbh = args.min_distance_bbh
x_max_bbh = args.max_distance_bbh

analysis = args.analysis
job_id = int(args.job_id)
type_inj = args.injection_type
# Read the argument to generate noise or injection 
if(analysis == 'injection'):
    path_folders = args.path_saving_data
elif(analysis == 'noise'):
    path_folders = args.path_saving_data

os.makedirs(path_folders, exist_ok=True)

# -------> Generate sineGaussian
    
num_injections = args.number_injections
for j in tqdm(range(num_injections)):
    if analysis == 'injection':
        seed = args.seed + job_id + j
    elif analysis == 'noise':
        seed = args.seed + job_id + j + 3153

    job_id_save = (args.number_injections * job_id) + j
    
    tsamples = int(80 * 2048) # generate 80 seconds of noise
    noise = pycbc.noise.noise_from_psd(tsamples, delta_t, psd, seed = seed)
    data = noise

    if(analysis == 'injection'):
        # -------> Generate the injections parameters
        m1 = random.uniform(args.min_m_bns,args.max_m_bns)
        m2 = random.uniform(args.min_m_bns,args.max_m_bns)
        for i in range(0,2):
            noise_zero = noise * 0
            m1 = m1 + random.uniform(-0.02, 0.02)
            m2 = m2 + random.uniform(-0.02, 0.02)
            mc = chirp_mass(m1, m2)
            spin1z = random.uniform(args.min_sz_bns, args.max_sz_bns)
            spin2z = random.uniform(args.min_sz_bns, args.max_sz_bns)
            d = generate_x2_distribution(x_min_bns, x_max_bns)
            distance_mc = chirp_distance(mc, d)
            # ------> Generate the injections
            hp, hc = get_td_waveform(approximant="SEOBNRv4_opt",
                                     mass1=m1,
                                     mass2=m2,
                                     spin1z = spin1z,
                                     spin2z = spin2z,
                                     delta_t=1 / 16384,
                                     f_lower=args.low_frequency_generating_injections,
                                     distance = distance_mc)
            # Let the signal begin in  the 61.3 -61.5  s window
            hp.start_time = 61 - hp.duration + random.uniform(-0.02,0.02)
            merge_time = hp.end_time
            hp = resample_to_delta_t(hp, 1.0/2048) # resample the signal to avoid over-computation 

            # -------> Inject bns in zero noise ts
            injection_ts = noise_zero.inject(hp) # inject the signal into this zero ts
            #injection_ts = injection_ts.inject(sineGaussian_ts)
            
            # ------> Calculate the optimal snr
            psd_est = interpolate(welch(noise), 1.0 / injection_ts.duration)
            optimal_snr = pycbc.filter.sigma(injection_ts, psd=psd_est,
                                             low_frequency_cutoff=args.low_frequency_generating_injections)

            injections_params = {"job_id": int(job_id_save),
                                 "distance": float(distance_mc),
                                 "m1": float(m1),
                                 "m2": float(m2),
                                 "s1z": float(spin1z),
                                 "s2z": float(spin2z),
                                 "merge_time1": float(merge_time),
                                 "seed": int(seed),
                                 "optimal_snr": float(optimal_snr),
                                 "detector": 'L1'}

            with open(path_folders + f'/injection_param_{job_id_save}.txt', 'a') as f:
                f.write(json.dumps(injections_params) + "\n")

            # -------> Inject the signal in noise
            data = data + injection_ts


                
    elif analysis == 'noise':
        injection_ts = noise_zero
        data = data + injection_ts

    # Inject a glitch
    activator = random.uniform(0, 1)
    threshold = args.glitch_threshold
    # -------> inject here the glitch If above threshold)
    if activator >= threshold:
        # -------> Generate sineGaussian                                                                                               
        f_0 = np.exp(np.random.uniform(np.log(20), np.log(2048)))
        q_factor = np.exp(np.random.uniform(np.log(3), np.log(400)))
        amplitude = random.uniform(1, 9) * (10**-22)
        phase = random.uniform(0, 6)
        time_jitter = random.uniform(-63,-0.5)
        sineGaussian_td = sineGaussian(noise.sample_times,
                                       t_0 = 63 + time_jitter, f_0 = f_0,
                                       q_factor = q_factor, amplitude = amplitude, phase = phase)
        sineGaussian_ts = pycbc.types.timeseries.TimeSeries(sineGaussian_td, delta_t)
        sineGaussian_params = {"job_id": job_id_save,
                                   "activator" : activator,
                                   "seed": seed,
                                   "f_0" : f_0,
                                   "q_factor" : q_factor,
                                   "amplitude" : amplitude,
                                   "phase" : phase,
                                   "time_jitter" : time_jitter,
                                   "detector": 'L1'}
        data = data + sineGaussian_ts # inject the sinegaussian glitch on 20% of data
        with open(path_folders + f'/sineGauss_param_{job_id_save}.txt', 'w') as f:
            f.write(json.dumps(sineGaussian_params) + "\n")
            
    if(analysis == 'injection'):
        name_file = f'/injection_{job_id_save}.gwf' # save the injection file                                                        
        frame.write_frame(path_folders  + name_file, 'L1', data)
        
    
    if analysis == 'noise':
        name_file = f'/noise_{job_id_save}.gwf'
        frame.write_frame(path_folders  + name_file, 'L1', data)
                        
end = time.perf_counter()
print(f"Duration: {end - start:.3f} seconds")
