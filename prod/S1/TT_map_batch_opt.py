#!/home/lorenzo-mobilia/.conda/envs/myenv3/bin/python  

import numpy as np
import pylab as pl
from tqdm import tqdm
from pycbc.frame import read_frame
from pycbc.filter import highpass_fir, matched_filter
from pycbc.waveform import get_fd_waveform
from pycbc.psd import welch, interpolate
import pycbc.vetoes
from urllib.request import urlretrieve
import pycbc.psd
from pycbc.psd import aLIGOZeroDetHighPower
import matplotlib.pyplot as plt
from PIL import ImageOps
import pandas as pd
import h5py
import matplotlib.colors as mplcolors
import scipy.ndimage

from PIL import Image
from PIL import PngImagePlugin, ImageOps
import glob
import pycbc.frame as frame
import sys
import ast
import argparse
import configparser
from pycbc.filter import matched_filter_core, make_frequency_series
from pycbc.vetoes.chisq import power_chisq_bins, power_chisq_at_points_from_precomputed
from pycbc.types import zeros
"""
------------------ TT MAP macro --------------
    This function will create the Time Template SNR time series map.
    It requires a template bank saved as frequency series and
    computes the matched filtering against the strech of data provided.
    The result is saved as a greyscale png image of 1024x512 dimension.
----------------------------------------------
"""

def parse_config(config_file):
    config = configparser.ConfigParser()
    config.read(config_file)

    g = config['general']

    args = argparse.Namespace(
        analysis                              = g.get('analysis'),
        path_to_time_series                   = g.get('path_to_time_series'),
        template_bank_frequency_domain        = g.get('template_bank_frequency_domain'),
        template_bank_parameters              = g.get('template_bank_parameteres'),
        number_of_time_series                 = g.getint('number_of_time_series'),
        number_of_templates                   = g.getint('number_of_templates'),
        low_frequency_cutoff_matched_filtering  = g.getfloat('low_frequency_cutoff_matched_filtering'),
        high_frequency_cutoff_matched_filtering = g.getfloat('high_frequency_cutoff_matched_filtering'),
        path_to_save_TT_map                   = g.get('path_to_save_TT_map'),
    )
    return args

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True, help="Path to config.ini")
parser.add_argument("--job-id", type=str, required=True, help="Job identifier")
cli = parser.parse_args()

args = parse_config(cli.config)
args.job_id = int(cli.job_id)


#-------------- Function to save the --------------------#
#-------------- image with metadata  -------------------#

def save_plot_with_metadata_in_memory(snrs, filename, metadata):
    """
    Renders a matplotlib plot directly into a 512x256 PNG image with metadata,
    without saving intermediate files.

    Args:
        times (array-like): X-axis values for the plot.
        mtots (array-like): Y-axis values for the plot.
        snrs (array-like): Z-axis values for the color intensity.
        filename (str): Name of the output PNG file.
        metadata (dict): Metadata to include in the PNG file.
    """
    # Downsample the array
    #snrs_downsampled = scipy.ndimage.zoom(snrs, (1/downsample_factor, 1/downsample_factor))
    #snrs_downsampled = snrs[::downsample_factor, ::downsample_factor]

    log_norm = mplcolors.LogNorm(vmin=0.1, vmax=100)
    normalized_snrs = log_norm(snrs)
    normalized_snrs = np.clip(normalized_snrs, 0, 1)
    normalized_snrs = (normalized_snrs * 255).astype(np.uint8)
    image = Image.fromarray(normalized_snrs, mode="L")
    resized_image = image.resize((512, 256), resample = Image.Resampling.BILINEAR)
    
    png_info = PngImagePlugin.PngInfo()
    for key, value in metadata.items():
        png_info.add_text(key, str(value))
    
    # Save the final image with metadata
    if filename.lower().endswith(".jpg") or filename.lower().endswith(".jpeg"):
        resized_image.save(filename, format="JPEG", quality=compression_quality)
    else:
        resized_image.save(filename, format="PNG", pnginfo=png_info, optimize=True)
    print(f"Image saved as {filename} with dimensions {image.size}, metadata: {metadata}, and size reduction applied.")

#--------------------------------#
    
#--------- Upload the data

# Read job and type of analysis
analysis = args.analysis
job_id = int(args.job_id)
num_data = args.number_of_time_series
len_freq = 81921

#with h5py.File(args.template_bank_parameters, 'r') as param_file:
#    mass1_all = param_file['mass1'][:]  # adjust key names after checking                                                  
#    mass2_all = param_file['mass2'][:]
#with h5py.File(args.template_bank_frequency_domain, 'r') as bank_file:
#    all_templates = bank_file['strain'][:, :len_freq] 

bank_file = h5py.File(args.template_bank_frequency_domain, 'r')
strain_ds = bank_file['strain']

for gwf in range(num_data):
    job_id_read_save = (num_data * job_id) + gwf
    print(f'processing id: {job_id_read_save}')
    # Path to your folder containing the files 
    if(analysis == 'injection'):
        path_folder = args.path_to_time_series
        file_name_read = f'/injection_{job_id_read_save}.gwf'
    elif(analysis == 'noise'):
        path_folder = args.path_to_time_series
        file_name_read = f'/noise_{job_id_read_save}.gwf'

    file_name = path_folder + file_name_read
    print(f'reading {file_name}')
    metadata = {}

    # Initialize the snrs and chisqs values.
    # The 0-axis is the number of templates in the template bank
    # The 1-axis is the time sampling where we will to save the snr time series
    # in this case is 4sec sampled at 2048Hz (4*2048 = 8192)
    num_templates = args.number_of_templates
    snrs = np.zeros((num_templates, 8192), dtype = np.float32)
    chisqs = np.zeros((num_templates, 8192), dtype = np.float32)

    # read the strech of data 
    data = frame.read_frame(file_name, channels='L1')
    # read the stretch of data
    data = frame.read_frame(file_name, channels='L1')
    psd  = interpolate(welch(data), 1.0 / data.duration)
    len_freq = len(data) // 2 + 1

    # --- FFT the data ONCE for the whole bank (not per template) ---
    stilde = make_frequency_series(data)
    corra  = zeros((len(stilde) - 1) * 2, dtype=stilde.dtype)

    num_templates = args.number_of_templates
    snrs = np.zeros((num_templates, 8192), dtype=np.float32)   # chisqs array no longer needed

    num_bins = 16
    SNR_CUT    = 4.5                       # SNR cut for hopless detection
    dof      = (num_bins * 2) - 2

    g_rwsnr, g_chisq = -1.0, float('nan')   # running global best, replaces full rwsnr map

    row_buf_c128 = np.empty(len_freq, dtype=np.complex128)

    for i in tqdm(range(num_templates)):
        row = num_templates - 1 - i
        strain_ds.read_direct(row_buf_c128, np.s_[i, :len_freq], np.s_[:])
        hp = pycbc.types.frequencyseries.FrequencySeries(
                row_buf_c128, delta_f=1.0 / 80, copy=False)
        if len(hp) != len(stilde):
            hp.resize(len(stilde))

        # ONE matched-filter core: unnormalized SNR + corr + norm
        snr, corr, snr_norm = matched_filter_core(
            hp, stilde, psd=psd,
            low_frequency_cutoff=args.low_frequency_cutoff_matched_filtering,
            high_frequency_cutoff=args.high_frequency_cutoff_matched_filtering,
            corr_out=corra)

        # SNR image row (the only array you save)
        snr_valid    = snr.crop(59, 17)
        snr_mag      = (np.abs(snr_valid.numpy()) * snr_norm).astype(np.float32)
        snrs[row, :] = snr_mag

        # chi-sq ONLY at the top-M SNR samples — direct time-shift, no IFFT
        loc      = np.nonzero(snr_mag > SNR_CUT)[0]
        if loc.size == 0:
            del snr, hp
            continue

        offset    = int(round(float(snr_valid.start_time - snr.start_time) / float(data.delta_t)))
        full_idx = (offset + loc).astype(np.uint32)
        snrv      = snr.numpy()[full_idx]                       # unnormalized complex

        bins      = power_chisq_bins(hp, num_bins, psd,
                        args.low_frequency_cutoff_matched_filtering,
                        args.high_frequency_cutoff_matched_filtering)
        chisq_raw = power_chisq_at_points_from_precomputed(
                        corr, snrv, snr_norm, bins, full_idx)
        rchisq    = np.asarray(chisq_raw) / dof

        snr_pts = snr_mag[loc]
        rw      = np.where(rchisq > 1,
                           snr_pts / ((1 + rchisq**3) / 2) ** (1.0 / 6),
                           snr_pts)
        b = int(rw.argmax())
        if rw[b] > g_rwsnr:
            g_rwsnr, g_chisq = float(rw[b]), float(rchisq[b])

        del snr, hp

    # --- global statistics (identical results to the full-map version) ---
    max_snr            = float(np.max(snrs))
    max_rwsnr          = g_rwsnr
    chisq_at_max_rwsnr = g_chisq    
    
    chisq_at_max_rwsnr = g_chisq
    # Save the metadata information related to this stretch of data
    if(analysis == 'injection'):
        with open(path_folder + f'/injection_param_{job_id_read_save}.txt', 'r') as f:
            for line in f:
                key, value = line.strip().split(':', 1)  # Split at the first column  
                metadata[key.strip()] = float(value.strip()) if '.' in value else value.strip()
            metadata['max_snr'] = max_snr
            metadata['max_rwsnr'] = max_rwsnr
            metadata['chisq'] = chisq_at_max_rwsnr
            print(metadata)
    else:
        with open(path_folder + f'/sineGauss_param_{job_id_read_save}.txt', 'r') as f:
            for line in f:
                key, value = line.strip().split(':', 1)  # Split at the first column
                metadata[key.strip()] = float(value.strip()) if '.' in value else value.strip()
            metadata['max_snr'] = max_snr
            metadata['max_rwsnr'] = max_rwsnr
            metadata['Label'] = 'Noise'
            metadata['chisq'] =	chisq_at_max_rwsnr
            print(metadata)

    # Save the TT-Map along with the metadata
    if(analysis == 'injection'):
        save_plot_with_metadata_in_memory(snrs, args.path_to_save_TT_map + f'/TT_map_SNR_{job_id_read_save}.png', metadata)
        #save_plot_with_metadata_in_memory(chisqs, f'/home/lorenzo-mobilia/bns_buster/data/S4/eccentricity/prod_TTMap/TT-maps-example/TT_map_chisq_{job_id}.png', metadata)
        #save_plot_with_metadata_in_memory(rwsnrs, f'/home/lorenzo-mobilia/bns_buster/data/S4/eccentricity/prod_TTMap/TT-maps-example/TT_map_rwsnr_{job_id}.png', metadata)
    else: 
        save_plot_with_metadata_in_memory(snrs, args.path_to_save_TT_map + f'/TTmap_{job_id_read_save}.png', metadata)
        #save_plot_with_metadata_in_memory(chisqs, f'/home/lorenzo-mobilia/bns_buster/data/Noise_chisqpowermap_50k_bns_bbh_McDistance_spin_uniform_correctChi_1024x512/ChiSqMap_{job_id}.png', metadata)
    

                                                                                               
    import resource

    usage = resource.getrusage(resource.RUSAGE_SELF)                                                                   
    mem_mb = usage.ru_maxrss / 1024 # ru_maxrss is in kB on Linux                                                      
    print(f"Peak memory usage (self-reported): {mem_mb:.2f} MB")  

