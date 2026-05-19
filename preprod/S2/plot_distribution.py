import os
import pandas as pd
import glob
import matplotlib.pyplot as plt
import numpy as np

# Define the folder path
folder_path = "/home/lorenzo-mobilia/EasyResNetPaper-v1/preprod/S2/injections/"

# Get all files matching the pattern
file_pattern = os.path.join(folder_path, "injection_param_*.txt")
files = glob.glob(file_pattern)

# Initialize an empty list to store data
data_list = []

# Loop through each file and read its contents
for file in files:
    with open(file, "r") as f:
        data = {}
        for line in f:
            key, value = line.strip().split(": ", 1)  # Split only on the first occurrence
            try:
                # Try converting to a float or int
                if "." in value or "e" in value:
                    data[key] = float(value)
                else:
                    data[key] = int(value)
            except ValueError:
                if "[" in value and "]" in value:  # Handle list values
                    value = value.strip("[]")
                    data[key] = [float(x) for x in value.split()]
                else:
                    data[key] = value  # Keep as string if it can't be converted
        data_list.append(data)

# Convert the list of dictionaries into a DataFrame
df = pd.DataFrame(data_list)
df['m_tot'] = df['m1'] + df['m2']
df['spin_eff'] = ( df['m1'] * df['s1z'] + df['m2'] * df['s2z'] ) / ( df['m_tot'])

save_path = '/home/lorenzo-mobilia/public_html/EasyResNetPaper-v1/S2/'

plt.scatter(df['distance'], df['optimal_snr'], s = 10)
plt.ylabel('opt_snr')
plt.xlabel('distance')
plt.xscale('log')
plt.savefig(save_path + f'distance_opt_snr.png')
plt.close()

plt.scatter(df['m_tot'], df['spin_eff'], s = 10)
plt.ylabel('$\chi_{eff}$')
plt.xlabel('Mtot')
plt.xscale('log')
plt.savefig(save_path + f'chi_eff_mtot1.png')
plt.close()

plt.hist(df['spin_eff'], bins = 100, density = True)
plt.ylabel('Density')
plt.xlabel('$\chi_{eff}$')
plt.savefig(save_path + f'chi_eff_histo.png')
plt.close()

plt.hist(df['s1z'], bins = 100, density = True)
plt.ylabel('Density')
plt.xlabel('$\chi_{1z}$')
plt.savefig(save_path + f'chi1z_histo.png')
plt.close()


plt.hist(df['s2z'], bins = 100, density = True)
plt.ylabel('Density')
plt.xlabel('$\chi_{2z}$')
plt.savefig(save_path + f'chi2z_histo.png')
plt.close()

distance_thresholds = np.arange(min(df['distance']),max(df['distance']), 0.5)
cumulative_distance = [np.sum(df['distance'] < distance) / len(df) for distance in distance_thresholds]

plt.step(distance_thresholds, cumulative_distance)
plt.yscale('log')
plt.xlabel('distance')
plt.ylabel('Fraction of events')
plt.savefig(save_path + 'fraction_distance.png')
plt.close()

opt_snr_thresholds = np.arange(min(df['optimal_snr']),max(df['optimal_snr']), 0.5)
cumulative_opt_snr = [np.sum(df['optimal_snr'] > snr) / len(df) for snr in opt_snr_thresholds]

plt.step(opt_snr_thresholds, cumulative_opt_snr)
plt.yscale('log')
plt.xlabel('opt_snr')
plt.ylabel('Fraction of events')
plt.savefig(save_path + 'fraction_opt_snr.png')
plt.close()

df['mc'] = (df['m1'] * df['m2'])**(3/5) / (df['m1'] + df['m2'])**(1/5)
plt.scatter(df['mc'], df['distance'], c = df['optimal_snr'] )
cbar = plt.colorbar()
cbar.set_label('optimal_snr')
plt.xlabel('mc')
plt.ylabel('distance')
plt.xscale('log')
plt.yscale('log')
plt.savefig(save_path + 'mc_distance_inj_prob.png')
plt.close()

plt.figure(figsize = (8,6))
counts_m1, bin_edges_m1 = np.histogram(df['m1'], bins=100, density=True)
plt.step(bin_edges_m1[:-1], counts_m1, where='mid', color='blue', linewidth=1.5, label="m1")
counts_m2, bin_edges_m2 = np.histogram(df['m2'], bins=50, density=True)
plt.step(bin_edges_m2[:-1], counts_m2, where='mid', color='red', linewidth=1.5, label="m2")

plt.xlabel('Mass')
plt.ylabel('Density')
plt.title('Masses distribution')
plt.yscale('log')
plt.legend()

plt.savefig(save_path + 'Masses_distribution.png')
plt.show()
