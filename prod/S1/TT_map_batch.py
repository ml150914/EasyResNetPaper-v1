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
for gwf in range(num_data):
    job_id_read_save = (num_data * job_id) + gwf
    print(f'processing id: {job_id_read_save}')
    # Path to your folder containing the files 
    if(analysis == 'injection'):
        path_folder = args.path_to_time_series
        file_name_read = f'/injection_{job_id_read_save}.gwf'
    elif(analysis == 'noise'):
        path_folder = ''
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
    # interpolate the psd
    psd = interpolate(welch(data), 1.0 / data.duration)
    # Compute the matched filtering for the whole bank (in frequency domain)
    len_freq = len(data) // 2 + 1
    snr_max = 0
    index_template_max_snr = 0
    best_template = None
    best_snr_series = None
    # parameters for the correct bin value
    with h5py.File(args.template_bank_parameters, 'r') as param_file:
        mass1_all = param_file['mass1'][:]  # adjust key names after checking
        mass2_all = param_file['mass2'][:]
    with h5py.File(args.template_bank_frequency_domain, 'r') as bank_file:
        strain_ds = bank_file['strain']
        for i in tqdm(range(num_templates)):
            # Read the template
            hp_np = strain_ds[i,:len_freq]
            # Resize if and change it to a FrequencySeries data-type
            #hp.resize(len(data) // 2 + 1)
            hp = pycbc.types.frequencyseries.FrequencySeries(hp_np, delta_f = 1/80)
        
            # Calculate the complex (two-phase) SNR
            snr = matched_filter(hp, data, psd=psd, low_frequency_cutoff=args.low_frequency_cutoff_matched_filtering, high_frequency_cutoff=args.high_frequency_cutoff_matched_filtering)
            
            # Remove regions corrupted by filter wraparound
            # Calculate the snr time-series in the 4s window around the merging
            snr = snr.crop(59, 17)
            # Consider the absolute value of the snr complex-time series
            snr = abs(snr)
            mass1 = float(mass1_all[i])
            mass2 = float(mass2_all[i])
            snrs[num_templates - 1 - i, :] = snr.numpy()  # Direct assignment
            num_bins = 16
            #num_bins_function = max(0.72*pycbc.pnutils.get_freq('fSEOBNRv4Peak',mass1,mass2,0,0)**0.7,11)
            #num_bins = min(max_number_of_bins, num_bins_function)
            chisq = pycbc.vetoes.power_chisq(hp, data, num_bins, psd = psd, low_frequency_cutoff=args.low_frequency_cutoff_matched_filtering, high_frequency_cutoff=args.high_frequency_cutoff_matched_filtering)
            
            chisq /= (num_bins * 2) - 2
            chisq = chisq.crop(59, 17)
            chisqs[num_templates - 1 - i, :] = chisq.numpy()
            
    # Compute now the rwsnr
    # Calculate the chisq
    

    rwsnrs = np.copy(snrs)
    mask = chisqs > 1
    rwsnrs[mask] = snrs[mask] / ((1 + chisqs[mask]**3) / 2) ** (1/6)
    
    # Find the maximum snr and rwsnr in the time series
    max_snr = float(np.max(snrs))
    max_rwsnr = float(np.max(rwsnrs))

    # Uncomment this to check the effect of the chisq test to the snr maximum
    # You should see that if a glitch is present the max snr is likely associated
    # to a template non-corresponding to the injection
    # while the rwsnr should
    
    indices = np.argwhere(np.isclose(rwsnrs, max_rwsnr))
    for idx in indices:
        idx = tuple(idx)
        print(f"Index: {idx}")
        print(f"  snr = {snrs[idx]}")
        print(f"  chisq = {chisqs[idx]}")
        print(f"  rwsnr = {rwsnrs[idx]}")
    
    idx_max_rwsnr = np.unravel_index(np.argmax(rwsnrs), rwsnrs.shape)
    chisq_at_max_rwsnr = float(chisqs[idx_max_rwsnr])
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
        save_plot_with_metadata_in_memory(snrs, args.path_to_save_TT_map + f'/TT_map_SNR_{job_id}.png', metadata)
        #save_plot_with_metadata_in_memory(chisqs, f'/home/lorenzo-mobilia/bns_buster/data/S4/eccentricity/prod_TTMap/TT-maps-example/TT_map_chisq_{job_id}.png', metadata)
        #save_plot_with_metadata_in_memory(rwsnrs, f'/home/lorenzo-mobilia/bns_buster/data/S4/eccentricity/prod_TTMap/TT-maps-example/TT_map_rwsnr_{job_id}.png', metadata)
    else: 
        save_plot_with_metadata_in_memory(snrs, args.path_to_save_TT_map + f'/TTmap_{job_id}.png', metadata)
        #save_plot_with_metadata_in_memory(chisqs, f'/home/lorenzo-mobilia/bns_buster/data/Noise_chisqpowermap_50k_bns_bbh_McDistance_spin_uniform_correctChi_1024x512/ChiSqMap_{job_id}.png', metadata)
    

                                                                                               
    import resource

    usage = resource.getrusage(resource.RUSAGE_SELF)                                                                   
    mem_mb = usage.ru_maxrss / 1024 # ru_maxrss is in kB on Linux                                                      
    print(f"Peak memory usage (self-reported): {mem_mb:.2f} MB")  

