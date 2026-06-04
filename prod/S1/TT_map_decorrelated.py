#!/usr/bin/env python3
"""
------------------ TT MAP (decorrelated, data-driven shifts) --------------

    WHY THE PN FORMULA ALONE DOES NOT WORK
    ----------------------------------------
    The total PN inspiral time range across the bank is ~43.5 s, but the
    cropped SNR window is only 4 s.  Rolling by the full PN t_coal
    difference wraps around ~11 times in np.roll, so the net shift visible
    inside the window is nearly zero — the image looks unchanged.

    CORRECT APPROACH: DATA-DRIVEN PEAK ALIGNMENT
    -----------------------------------------------
    For each template k, we measure the sample index of the SNR peak
    directly from argmax(snr[k, :]) inside the cropped 4 s window.
    The shift is then:

        shift[k] = peak_sample[k] - peak_sample[ref_idx]

    This is valid because:
      - It works regardless of where the window is positioned relative
        to the merger time.
      - It captures PN timing differences *modulo the window length*,
        which is exactly what is visible in the image.
      - For injections, ref_idx = loudest template (closest to true Mc),
        so the brightest ridge column ends up centred in the image.
      - For noise, ref_idx = geometric-mean Mc template (fixed anchor).

    After rolling, the matrix is transposed:
        [num_templates, time]  -->  [time, num_templates]
    so time is on the vertical axis and chirp mass on the horizontal axis.

    Two images are saved per file:
        *_full.png  : resized to 512x256  (same footprint as original)
        *_zoom.png  : native-resolution crop around the SNR peak
---------------------------------------------------------------------------
"""

import numpy as np
import argparse
import configparser
import os
import h5py
import resource

import matplotlib.colors as mplcolors
from PIL import Image, PngImagePlugin
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

import pycbc.frame as frame
from pycbc.psd import welch, interpolate


# ------------------------------------------------------------------ #
#  Zoom window (samples in the full-resolution decorrelated array)     #
# ------------------------------------------------------------------ #
ZOOM_HALF_TIME  = 512*4    # ±512 time samples  = ±0.25 s at 2048 Hz
ZOOM_HALF_TEMPL = 256*4    # ±256 template bins


# ------------------------------------------------------------------ #
#  Config                                                              #
# ------------------------------------------------------------------ #

def parse_config(config_file):
    config = configparser.ConfigParser()
    config.read(config_file)
    g = config['general']
    return argparse.Namespace(
        analysis                                = g.get('analysis'),
        path_to_time_series                     = g.get('path_to_time_series'),
        template_bank_frequency_domain          = g.get('template_bank_frequency_domain'),
        template_bank_parameters                = g.get('template_bank_parameteres'),
        number_of_time_series                   = g.getint('number_of_time_series'),
        number_of_templates                     = g.getint('number_of_templates'),
        low_frequency_cutoff_matched_filtering  = g.getfloat('low_frequency_cutoff_matched_filtering'),
        high_frequency_cutoff_matched_filtering = g.getfloat('high_frequency_cutoff_matched_filtering'),
        path_to_save_TT_map                     = g.get('path_to_save_TT_map'),
    )


# ------------------------------------------------------------------ #
#  Per-template matched filter worker                                  #
# ------------------------------------------------------------------ #

def process_template(args_tuple):
    (i, hp_np, data_np, data_delta_t, data_epoch,
     psd_np, psd_delta_f, low_f, high_f) = args_tuple

    from pycbc.types import FrequencySeries, TimeSeries
    import pycbc.vetoes
    from pycbc.filter import matched_filter

    hp   = FrequencySeries(hp_np, delta_f=1.0 / 80)
    data = TimeSeries(data_np, delta_t=data_delta_t, epoch=data_epoch)
    psd  = FrequencySeries(psd_np, delta_f=psd_delta_f)

    snr = matched_filter(hp, data, psd=psd,
                         low_frequency_cutoff=low_f,
                         high_frequency_cutoff=high_f)
    snr = abs(snr.crop(59, 17))

    chisq = pycbc.vetoes.power_chisq(hp, data, 16, psd=psd,
                                      low_frequency_cutoff=low_f,
                                      high_frequency_cutoff=high_f)
    chisq /= (16 * 2) - 2
    chisq  = chisq.crop(59, 17)

    return i, snr.numpy(), chisq.numpy()


# ------------------------------------------------------------------ #
#  Reference index selection                                           #
# ------------------------------------------------------------------ #

def select_reference_index(analysis, snrs, mass1_all, mass2_all):
    """
    injection : loudest template (data-driven, closest to true Mc).
    noise     : geometric-mean Mc template (fixed, consistent layout).
    """
    if analysis == 'injection':
        ref_idx = int(np.argmax(snrs.max(axis=1)))
        mc_ref  = ( (mass1_all[ref_idx] * mass2_all[ref_idx])**(3/5)
                   / (mass1_all[ref_idx] + mass2_all[ref_idx])**(1/5) )
        print(f"[injection] ref_idx={ref_idx}  Mc_ref={mc_ref:.3f} Msun  "
              f"peak_SNR={snrs[ref_idx].max():.2f}")
    else:
        mc_all  = ( (mass1_all * mass2_all)**(3/5)
                   / (mass1_all + mass2_all)**(1/5) )
        mc_geom = np.exp(0.5*(np.log(mc_all.min()) + np.log(mc_all.max())))
        ref_idx = int(np.argmin(np.abs(mc_all - mc_geom)))
        print(f"[noise] ref_idx={ref_idx}  Mc_ref={mc_all[ref_idx]:.3f} Msun  "
              f"(geom-mean Mc={mc_geom:.3f})")
    return ref_idx


# ------------------------------------------------------------------ #
#  Data-driven decorrelation + transpose                               #
# ------------------------------------------------------------------ #

def decorrelate_time_template(snrs, ref_idx, sample_rate):
    """
    Measure the SNR peak sample for each template directly from the data,
    then roll each row so all peaks align with the reference template's peak.

        shift[k] = argmax(snr[k,:]) - argmax(snr[ref_idx,:])

    This is correct even when the total PN time range >> window length,
    because it operates entirely within the cropped 4 s window.

    After rolling, transpose: [templates, time] --> [time, templates]

    Returns
    -------
    snrs_rotated : shape (time_samples, num_templates)
    shifts       : int array, shape (num_templates,)
    peak_samples : int array, shape (num_templates,)  -- raw peak positions
    """
    # Peak sample for each template (argmax along time axis)
    peak_samples = np.argmax(snrs, axis=1)           # shape: (num_templates,)
    ref_peak     = peak_samples[ref_idx]

    shifts       = peak_samples - ref_peak            # positive = peaks later
    snrs_decorr  = np.empty_like(snrs)

    for k in range(snrs.shape[0]):
        # Roll left by shifts[k]: moves peak of template k
        # to the same column as the reference peak
        snrs_decorr[k, :] = np.roll(snrs[k, :], -int(shifts[k]))

    print(f"Peak sample of reference (idx={ref_idx}): {ref_peak}")
    print(f"Shift range: [{shifts.min()}, {shifts.max()}] samples  "
          f"= [{shifts.min()/sample_rate*1e3:.1f}, "
          f"{shifts.max()/sample_rate*1e3:.1f}] ms")

    return snrs_decorr, shifts, peak_samples   # (time, templates)


# ------------------------------------------------------------------ #
#  Log-normalise to uint8                                              #
# ------------------------------------------------------------------ #

def log_normalise(arr_2d, vmin=0.1, vmax=100):
    log_norm   = mplcolors.LogNorm(vmin=vmin, vmax=vmax)
    normalised = log_norm(arr_2d)
    normalised = np.clip(normalised, 0, 1)
    return (normalised * 255).astype(np.uint8)


# ------------------------------------------------------------------ #
#  Save full map  (resized to 512x256)                                 #
# ------------------------------------------------------------------ #

def save_full_map(snrs_2d, filename, metadata):
    u8   = log_normalise(snrs_2d)
    img  = Image.fromarray(u8, mode="L")
    img  = img.resize((512, 256), resample=Image.Resampling.BILINEAR)
    info = PngImagePlugin.PngInfo()
    for k, v in metadata.items():
        info.add_text(str(k), str(v))
    img.save(filename, format="PNG", pnginfo=info, optimize=True)
    print(f"Full map -> {filename}  size={img.size}")


# ------------------------------------------------------------------ #
#  Save zoom crop  (native resolution, no resize)                      #
# ------------------------------------------------------------------ #

def save_zoom_map(snrs_2d, filename, metadata,
                  zoom_half_time=ZOOM_HALF_TIME,
                  zoom_half_templ=ZOOM_HALF_TEMPL):
    """
    Crop ±zoom_half_time rows and ±zoom_half_templ columns around the
    global SNR peak in the decorrelated array and save at native resolution.
    """
    n_time, n_templ = snrs_2d.shape
    peak_t, peak_k  = np.unravel_index(np.argmax(snrs_2d), snrs_2d.shape)

    t0 = max(0,       peak_t - zoom_half_time)
    t1 = min(n_time,  peak_t + zoom_half_time)
    k0 = max(0,       peak_k - zoom_half_templ)
    k1 = min(n_templ, peak_k + zoom_half_templ)

    crop = snrs_2d[t0:t1, k0:k1]
    u8   = log_normalise(crop)
    img  = Image.fromarray(u8, mode="L")   # native resolution

    meta = dict(metadata)
    meta.update({
        'zoom_peak_time_sample': int(peak_t),
        'zoom_peak_templ_idx':   int(peak_k),
        'zoom_t0': t0, 'zoom_t1': t1,
        'zoom_k0': k0, 'zoom_k1': k1,
        'zoom_half_time':  zoom_half_time,
        'zoom_half_templ': zoom_half_templ,
    })
    info = PngImagePlugin.PngInfo()
    for k, v in meta.items():
        info.add_text(str(k), str(v))
    img.save(filename, format="PNG", pnginfo=info, optimize=True)
    print(f"Zoom map -> {filename}  crop=({t0}:{t1}, {k0}:{k1})  "
          f"size={img.size}  peak=({peak_t},{peak_k})")


# ------------------------------------------------------------------ #
#  Main                                                                #
# ------------------------------------------------------------------ #

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--job-id", type=str, required=True)
    cli = parser.parse_args()

    args        = parse_config(cli.config)
    args.job_id = int(cli.job_id)

    analysis      = args.analysis
    job_id        = args.job_id
    N_WORKERS     = 4
    SAMPLE_RATE   = 2048
    TIME_SAMPLES  = 8192
    num_templates = args.number_of_templates

    # ---- Load bank ---------------------------------------------- #
    with h5py.File(args.template_bank_parameters, 'r') as pf:
        mass1_all = pf['mass1'][:num_templates]
        mass2_all = pf['mass2'][:num_templates]

    with h5py.File(args.template_bank_frequency_domain, 'r') as bf:
        all_templates = bf['strain'][:]

    # ---- File paths --------------------------------------------- #
    if analysis == 'injection':
        file_name  = os.path.join(args.path_to_time_series,
                                  f'injection_{job_id}.gwf')
        param_path = os.path.join(args.path_to_time_series,
                                  f'injection_param_{job_id}.txt')
        out_full   = os.path.join(args.path_to_save_TT_map,
                                  f'TT_map_decorr_{job_id}_full.png')
        out_zoom   = os.path.join(args.path_to_save_TT_map,
                                  f'TT_map_decorr_{job_id}_zoom.png')
    else:
        file_name  = os.path.join(args.path_to_time_series,
                                  f'noise_{job_id}.gwf')
        param_path = os.path.join(args.path_to_time_series,
                                  f'sineGauss_param_{job_id}.txt')
        out_full   = os.path.join(args.path_to_save_TT_map,
                                  f'TTmap_decorr_{job_id}_full.png')
        out_zoom   = os.path.join(args.path_to_save_TT_map,
                                  f'TTmap_decorr_{job_id}_zoom.png')

    print(f'Reading {file_name}')

    # ---- Read data & PSD ---------------------------------------- #
    data     = frame.read_frame(file_name, channels='L1')
    psd      = interpolate(welch(data), 1.0 / data.duration)
    len_freq = len(data) // 2 + 1

    # ---- Matched filtering -------------------------------------- #
    snrs   = np.zeros((num_templates, TIME_SAMPLES), dtype=np.float32)
    chisqs = np.zeros((num_templates, TIME_SAMPLES), dtype=np.float32)

    task_args = [
        (i,
         all_templates[i, :len_freq],
         np.array(data), data.delta_t, float(data.start_time),
         np.array(psd),  psd.delta_f,
         args.low_frequency_cutoff_matched_filtering,
         args.high_frequency_cutoff_matched_filtering)
        for i in range(num_templates)
    ]

    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        for i, snr_arr, chisq_arr in tqdm(
                executor.map(process_template, task_args),
                total=num_templates, desc="Matched filtering"):
            snrs[i, :]   = snr_arr
            chisqs[i, :] = chisq_arr

    # ---- Reweighted SNR ----------------------------------------- #
    rwsnrs = np.where(chisqs > 1,
                      snrs / ((1 + chisqs**3) / 2)**(1.0/6.0),
                      snrs)

    # ---- Reference index (after MF) ----------------------------- #
    ref_idx = select_reference_index(analysis, snrs, mass1_all, mass2_all)

    # ---- Data-driven decorrelation + transpose ------------------ #
    snrs_rotated, shifts, peak_samples = decorrelate_time_template(
        snrs, ref_idx, SAMPLE_RATE
    )
    # snrs_rotated: shape (TIME_SAMPLES, num_templates)

    # ---- Statistics --------------------------------------------- #
    max_snr   = float(np.max(snrs))
    max_rwsnr = float(np.max(rwsnrs))
    idx_max   = np.unravel_index(np.argmax(rwsnrs), rwsnrs.shape)
    chisq_max = float(chisqs[idx_max])

    # ---- Metadata ----------------------------------------------- #
    metadata = {}
    with open(param_path, 'r') as f:
        for line in f:
            key, value = line.strip().split(':', 1)
            metadata[key.strip()] = (float(value.strip())
                                     if '.' in value else value.strip())
    metadata['max_snr']   = max_snr
    metadata['max_rwsnr'] = max_rwsnr
    metadata['chisq']     = chisq_max
    metadata['ref_idx']   = ref_idx
    metadata['ref_mc']    = float(
        (mass1_all[ref_idx] * mass2_all[ref_idx])**(3/5)
        / (mass1_all[ref_idx] + mass2_all[ref_idx])**(1/5)
    )
    if analysis == 'noise':
        metadata['label'] = 'Noise'
    print(metadata)

    # ---- Save full + zoom --------------------------------------- #
    save_full_map(snrs_rotated, out_full, metadata)
    save_zoom_map(snrs_rotated, out_zoom, metadata)

    usage  = resource.getrusage(resource.RUSAGE_SELF)
    mem_mb = usage.ru_maxrss / 1024
    print(f"Peak memory: {mem_mb:.2f} MB")
