import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from PIL import Image

def get_png_files(folder):
    with os.scandir(folder) as it:
        return [e.path for e in it if e.name.endswith(".png") and e.is_file()]

def read_png_metadata(image_path):
    with Image.open(image_path) as img:
        return {k: _cast(v) for k, v in img.info.items()}

def _cast(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return value
    
class MyFormatter(
    argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter
):
    pass
    
parser = argparse.ArgumentParser(
    prog="snr_bkg_study",
    description="Runs a SNR distribution study for Gaussian Noise bkg",
    epilog=f"""add_help=""",
    formatter_class=MyFormatter,
)

parser.add_argument("--input_file_inj", help=f"name of the folder to read the injection INPUT file", type=str)
parser.add_argument("--input_file_noise", help=f"name of the folder to read the noise INPUT file", type=str)
parser.add_argument("--output_plots", help=f"name of the folder to plot OUT file", type=str)

args = parser.parse_args()

folder_path_inj = args.input_file_inj
folder_path_noise = args.input_file_noise

files_inj = get_png_files(args.input_file_inj)
files_noise = get_png_files(args.input_file_noise)

with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
    inj_df   = pd.DataFrame(executor.map(read_png_metadata, files_inj,   chunksize=500))
    noise_df = pd.DataFrame(executor.map(read_png_metadata, files_noise, chunksize=500))


inj_df = inj_df.astype(float, errors = 'ignore')
noise_df = noise_df.astype(float, errors = 'ignore')

#print(noise_df['max_snr'])

inj_snr = inj_df['max_snr'].to_numpy()
noise_snr = noise_df['max_snr'].to_numpy()

#print(inj_snr.min())
#print(inj_snr.max())
print(noise_snr.min())
#print(noise_snr.max())

thrs = np.linspace(min(noise_snr.min(), inj_snr.min()),
                   max(noise_snr.max(), inj_snr.max()),
                   num=1000)

print(thrs)

inj_above_threshold = [np.sum(inj_snr > thr) / len(inj_snr) for thr in thrs]
noise_above_threshold = [np.sum(noise_snr > thr) / len(noise_snr) for thr in thrs]


plt.plot(noise_above_threshold, thrs, label='noise', c='r', linewidth=3.0)
plt.plot(inj_above_threshold, thrs, label='inj', c='b', linewidth=2.0)
plt.xlabel('SNR treshold')
plt.ylabel('#')
plt.yscale('log')
plt.xscale('log')
#fig, ax = plt.subplots()

#inj_df['max_snr'].plot.hist(ax=ax, cumulative=True, density=True, bins=100, alpha=0.5, label='Injections')
#noise_df['max_snr'].plot.hist(ax=ax, cumulative=True, density=True, bins=100, alpha=0.5, label='Noise')

#ax.set_xlabel('SNR')
#ax.set_ylabel('Cumulative Density')
#ax.set_title('SNR Cumulative Distribution')
#ax.legend()
plt.savefig(os.path.join(args.output_plots, 'SNR_histo.png'), dpi=300)
plt.close()

