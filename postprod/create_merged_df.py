import os
import glob
import pandas as pd
from PIL import Image

"""--------------- create_fd.py -------------

    Collects the output of train.py (CNN test-results CSV) and the
    corresponding images' PNG metadata, and creates a merged final dataset,
    split into injections and noise.
"""

# --- EDIT THESE PATHS ---
folder_path_images = '/home/lorenzo-mobilia/EasyResNetPaper-v1/prod/S4/Collective-Dataset/dataset_collected_70k/test'
csv_test_results = '/home/lorenzo-mobilia/public_html/EasyResNetPaper-v1/S4/results_70k/test_scores.csv'
output_dir = '/home/lorenzo-mobilia/public_html/EasyResNetPaper-v1/S4/results_70k/'                                     # where to save the output CSVs
# -------------------------

files_images = glob.glob(os.path.join(folder_path_images, '*.png'))
print(f'Found {len(files_images)} images in {folder_path_images}')


def read_png_metadata(image_path):
    """Extracts metadata added using PngInfo from a PNG file."""
    with Image.open(image_path) as img:
        meta = {key: img.info[key] for key in img.info.keys()}
    meta['file'] = os.path.normpath(image_path)  # store the PATH, not the Image object
    return meta


metadata_list_inj = []
metadata_list_noise = []

print('-------------- Read and save Metadata ---------------')
for pics in files_images:
    try:
        meta = read_png_metadata(pics)
    except Exception as e:
        print(f"Error reading {pics}: {e}")
        continue  # skip this file entirely instead of reusing stale metadata

    if 'distance' in meta:
        metadata_list_inj.append(meta)
    else:
        metadata_list_noise.append(meta)

df_metadata_inj = pd.DataFrame(metadata_list_inj)
df_metadata_noise = pd.DataFrame(metadata_list_noise)

print(f'# Injections: {len(df_metadata_inj)}')
print(f'# Noise: {len(df_metadata_noise)}')

# --- Load the CNN test results ---
df_test_results = pd.read_csv(csv_test_results)
df_test_results['file'] = df_test_results['file'].apply(lambda p: os.path.normpath(p.strip()))

# --- Avoid duplicate columns on merge ---
# Both the PNG metadata and the CNN results CSV carry ground-truth fields like
# distance/activator/optimal_snr/max_snr. Keep those from the metadata (the
# source of truth) and only pull the CNN-specific columns from the results CSV.
if 'max_rwsnr' in df_metadata_inj.columns:
    cnn_only_cols = ['file', 'Inj_Prob', 'Noise_Prob', 'Predicted_label', 'Label']
    df_test_results_slim = df_test_results[[c for c in cnn_only_cols if c in df_test_results.columns]]
else:
    cnn_only_cols = ['file', 'Inj_Prob', 'Noise_Prob', 'Predicted_label', 'Label', 'max_rwsnr', 'chisq']
    df_test_results_slim = df_test_results[[c for c in cnn_only_cols if c in df_test_results.columns]]

# --- Merge metadata with CNN results ---
# how='left' keeps every image's metadata even if a CNN result is missing.
df_inj_final = pd.merge(df_metadata_inj, df_test_results_slim, on='file', how='left')
df_noise_final = pd.merge(df_metadata_noise, df_test_results_slim, on='file', how='left')

print(f'Merged injections: {len(df_inj_final)} '
      f'(unmatched: {df_inj_final["Inj_Prob"].isna().sum() if "Inj_Prob" in df_inj_final else "n/a"})')
print(f'Merged noise: {len(df_noise_final)} '
      f'(unmatched: {df_noise_final["Inj_Prob"].isna().sum() if "Inj_Prob" in df_noise_final else "n/a"})')

# --- Fix dtypes ---
# PNG metadata (img.info) always comes back as strings, even for numeric fields.
# Convert the known-numeric columns so downstream analysis/plotting doesn't break.
numeric_cols = ['max_snr', 'distance', 'm1', 'm2', 'merge_time', 'optimal_snr',
                'seed', 'f_0', 'q_factor', 'amplitude', 'phase', 'time_jitter',
                'Inj_Prob', 'Noise_Prob', 'max_rwsnr', 'chisq']

for df_ in (df_inj_final, df_noise_final):
    for col in numeric_cols:
        if col in df_.columns:
            df_[col] = pd.to_numeric(df_[col], errors='coerce')

# --- Save results ---
df_inj_final.to_csv(os.path.join(output_dir, 'dfInj_test_dataset.csv'), sep=',', index=False)
df_noise_final.to_csv(os.path.join(output_dir, 'dfNoise_test_dataset.csv'), sep=',', index=False)

print('Done. Saved dfInj_test_dataset.csv and dfNoise_test_dataset.csv')
