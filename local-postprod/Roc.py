import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
})

def CDF(data: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """
    Computes the Cumulative Density Function (CDF) for a population t (noise or
    injections), evaluated at a set of thresholds r_hat:

        CDF_t(r_hat) = (1 / N_t) * sum_i theta(r_i - r_hat)

    where theta is the Heaviside step function, r_i are the N_t samples of
    population t, and r_hat is the threshold at which the CDF is evaluated.
    Equivalently, this is the fraction of samples in `data` strictly greater
    than each threshold.

    :param data: 1D array of samples for population t (e.g. Inj_Prob, max_snr, ...)
    :param thresholds: 1D array of threshold values (r_hat) at which to evaluate the CDF
    :return: 1D array (same length as `thresholds`) with CDF_t(r_hat) for each threshold
    """
    data = np.asarray(data)
    thresholds = np.asarray(thresholds)

    # theta(r_i - r_hat) = 1 if r_i > r_hat, else 0
    # shape: (len(thresholds), len(data)) via broadcasting, then average over data axis
    above_threshold = data[None, :] > thresholds[:, None]

    return above_threshold.mean(axis=1)

def plot_CDF(df1, df2, column, var, label1='Noise', label2='Injections',
             save=True, thr=0.0001, colors=('b', 'r'), linewidth=1.5):
    """
    Plots CDF_t(r_hat) = (1/N_t) * sum_i theta(r_i - r_hat) as a function of the
    threshold r_hat, for two populations t (e.g. noise and injections), on the
    specified column.

    :param df1: DataFrame for population 1 (e.g. noise)
    :param df2: DataFrame for population 2 (e.g. injections)
    :param column: column name to compute the CDF over (e.g. 'Inj_Prob', 'max_snr')
    :param var: label used in the saved filename/CSV
    :param label1: legend label for population 1
    :param label2: legend label for population 2
    :param save: whether to save the underlying threshold/CDF values to CSV
    :param thr: threshold step size used to build the threshold grid
    :param colors: tuple of two colors, one per population
    :param linewidth: line width for both curves
    :return: DataFrame with columns 'threshold', f'CDF_{label1}', f'CDF_{label2}'
    """
    if column not in df1.columns:
        raise ValueError(f"Column '{column}' not found in the DataFrame.")

    data1 = df1[column].dropna()
    data2 = df2[column].dropna()

    thresholds = np.arange(min(data1.min(), data2.min()),
                            max(data1.max(), data2.max()), thr)

    cdf1 = CDF(data1, thresholds)
    cdf2 = CDF(data2, thresholds)

    cdf_df = pd.DataFrame({
        'threshold': thresholds,
        f'CDF_{label1}': cdf1,
        f'CDF_{label2}': cdf2,
    })

    if save:
        cdf_df.to_csv(path_data + f'/{var}_CDF_{column}.csv', index=False)

    plt.plot(thresholds, cdf1, label=label1, c=colors[0], lw=linewidth)
    plt.plot(thresholds, cdf2, label=label2, c=colors[1], lw=linewidth)

    return cdf_df

def save_cdf(label, title, column):
    plt.title(f'CDF - {title}')
    plt.xlabel('CNN output')
    plt.ylabel(r'$\mathrm{CDF}_t$')
    plt.grid(True)
    plt.legend()
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(path_data + f'CDF_{label}_{column}.png', dpi=300)
    plt.close()

def RoC(df1, df2, column, var, label, save=True, thr=0.0001, color='b', linewidth=1.5, ax=None):
    if ax is None:
        ax = plt.gca()
    if column not in df1.columns:
        raise ValueError(f"Column '{column}' not found in the DataFrame.")

    data1 = df1[column].dropna()
    data2 = df2[column].dropna()
    thresholds = np.arange(min(data1), max(data1), thr)

    count_above_threshold1 = [np.sum(data1 > t) / len(data1) for t in thresholds]
    count_above_threshold2 = [np.sum(data2 > t) / len(data2) for t in thresholds]

    cumulative_df = pd.DataFrame({
        'threshold': thresholds,
        'count_above_threshold1': count_above_threshold1,
        'count_above_threshold2': count_above_threshold2
    })

    if save:
        cumulative_df.to_csv(path_data + f'/{var}_RoC_{column}.csv', index=False)

    ax.plot(cumulative_df['count_above_threshold1'], cumulative_df['count_above_threshold2'], label=label, c=color, lw=linewidth)
    return cumulative_df

def scatter_distance(df, title, ax, fig):
    sc = ax.scatter(df['distance'], df['optimal_snr'], c=df['Inj_Prob'], cmap='copper', s=1.2)
    ax.set_xlabel('Distance [Mpc]')
    ax.set_ylabel(r'$\rho_\mathrm{opt}$')
    ax.set_xscale('log')
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label('Injection Probability')
    ax.set_title(title)

def save_roc(label, title):
    plt.title(f'Cumulative Distribution (Counts Above Threshold) - {title}')
    plt.xlabel('FAP')
    plt.ylabel('TAP')
    plt.grid(True)
    plt.legend()
    plt.yscale('log')
    plt.xscale('log')
    #plt.ylim([0.1, 1.1])                                                                                 
    plt.tight_layout()
    plt.savefig(path_data + f'Roc_Superimposed_{label}.png', dpi = 300)
    plt.close()

def plot_roc_and_scatter(df_noise, df_inj, tag_title, save_label):
    """
    Single figure with:
      - top row: ROC curves (Inj_Prob, max_snr, max_rwsnr)
      - bottom row: distance vs optimal SNR scatter

    :param df_noise: noise DataFrame (already sampled to match n_detected, if relevant)
    :param df_inj: injection DataFrame (already filtered to the tag of interest, if relevant)
    :param tag_title: human-readable title, e.g. 'Precessing'
    :param save_label: filename-safe label, e.g. 'S4_precessing'
    """
    fig, (ax_roc, ax_scatter) = plt.subplots(2, 1, figsize=(7, 10))

    # --- Top row: ROC ---
    RoC(df_noise, df_inj, 'Inj_Prob', 'Roc', 'CNN output',
         thr=0.00001, color=colors[0], linewidth=5.5, ax=ax_roc)
    RoC(df_noise, df_inj, 'max_snr', 'Roc', r'max $\rho$',
         thr=0.005, color=colors[1], linewidth=3.75, ax=ax_roc)
    if 'max_rwsnr' in df_inj.columns:
        RoC(df_noise, df_inj, 'max_rwsnr', 'Roc', r'max $\rho_\mathrm{rw}$',
             thr=0.0005, color=colors[2], linewidth=2.5, ax=ax_roc)

    ax_roc.set_title(f'Cumulative Distribution (Counts Above Threshold) - {tag_title}')
    ax_roc.set_xlabel('FAP')
    ax_roc.set_ylabel('TAP')
    ax_roc.grid(True)
    ax_roc.legend()
    ax_roc.set_yscale('log')
    ax_roc.set_xscale('log')
    ax_roc.set_ylim([0.1, 1.1])

    # --- Bottom row: scatter ---
    if 'distance' in df_inj.columns:
        scatter_distance(df_inj, tag_title, ax_scatter, fig)
    else:
        ax_scatter.set_visible(False)

    plt.tight_layout()
    plt.savefig(path_data + f'RocScatter_{save_label}.png', dpi=300)
    plt.close(fig)

simulation = 'S1'

file_inj = '/Users/lorenzomobilia/Desktop/Lavoro/PhDUrbino/IJC_lab/EasyResNetPaper-v1-1/local-postprod/csv_downloads/dfInj_test_dataset_S1.csv'
file_noise = '/Users/lorenzomobilia/Desktop/Lavoro/PhDUrbino/IJC_lab/EasyResNetPaper-v1-1/local-postprod/csv_downloads/dfNoise_test_dataset_S1.csv'
path_data = '/Users/lorenzomobilia/Desktop/Lavoro/PhDUrbino/IJC_lab/EasyResNetPaper-v1-1/local-postprod/plots/S1/'

df_inj = pd.read_csv(file_inj)
df_noise = pd.read_csv(file_noise)
df_inj.dropna(how='all', axis=1, inplace=True)
df_noise.dropna(how='all', axis=1, inplace=True)

colors = ['#0072B2', '#E69F00', '#009E73', "#9E0032",  "#699E00"]

plot_CDF(df_noise, df_inj, 'Inj_Prob', simulation, label1='Noise', label2='Injections',
          thr=0.00001, colors=(colors[3], colors[4]), linewidth=2.5)
save_cdf(simulation, simulation, 'Inj_Prob')

RoC(df_noise, df_inj, 'Inj_Prob', 'Roc', 'CNN output', thr = 0.00001, color = colors[0], linewidth=1.5)
RoC(df_noise, df_inj, 'max_snr', 'Roc', r'max $\rho$', thr = 0.005, color = colors[1], linewidth= 3.75)

if 'max_rwsnr' in df_inj.columns:
    RoC(df_noise, df_inj, 'max_rwsnr', 'Roc', r'max $\rho_\mathrm{rw}$', thr = 0.0005, color = colors[2], linewidth= 2.5)
else:
    print('max_rwsnr not present')
# Final plot settings                                                                                     \
                                                                                                           
save_roc(simulation, simulation)

if 'distance' in df_inj.columns:
    fig, ax = plt.subplots()
    scatter_distance(df_inj, simulation, ax, fig)
    plt.savefig(path_data + f'Distance_opt_snr_{simulation}.png', dpi=300)
    plt.close(fig)


# S4 analysis 
if 'tag' in df_inj.columns:
    df_inj_precessing   = df_inj[df_inj['tag'] == 'Precessing']
    df_inj_superimposed = df_inj[df_inj['tag'] == 'Superimposed']
    df_inj_HoM          = df_inj[df_inj['tag'] == 'HoM']
    df_inj_Eccentricity = df_inj[df_inj['tag'] == 'Eccentricity']
    df_inj_ExtremeSpin  = df_inj[df_inj['tag'] == 'ExtremeSpin']

    n_precessing_detected   = len(df_inj_precessing[df_inj_precessing['Inj_Prob'] > 0.5])
    n_superimposed_detected = len(df_inj_superimposed[df_inj_superimposed['Inj_Prob'] > 0.5])
    n_HoM_detected          = len(df_inj_HoM[df_inj_HoM['Inj_Prob'] > 0.5])
    n_Eccentricity_detected = len(df_inj_Eccentricity[df_inj_Eccentricity['Inj_Prob'] > 0.5])
    n_ExtremeSpin_detected  = len(df_inj_ExtremeSpin[df_inj_ExtremeSpin['Inj_Prob'] > 0.5])

    tag_configs = [
    (df_inj_precessing,   n_precessing_detected,   'Precessing'),
    (df_inj_superimposed, n_superimposed_detected, 'Superimposed'),
    (df_inj_HoM,          n_HoM_detected,           'HoM'),
    (df_inj_Eccentricity, n_Eccentricity_detected,  'Eccentricity'),
    (df_inj_ExtremeSpin,  n_ExtremeSpin_detected,   'Extreme Spins'),
]

    n_cols = len(tag_configs)
    fig, axes = plt.subplots(2, n_cols, figsize=(5 * n_cols, 10))
    # axes[0, j] = ROC for tag j (top row)
    # axes[1, j] = scatter for tag j (bottom row)

    for j, (df_inj_tag, n_detected, tag_title) in enumerate(tag_configs):
        ax_roc = axes[0, j]
        ax_scatter = axes[1, j]

        df_noise_sampled = df_noise.sample(n=n_detected)

        # --- top row: ROC ---
        RoC(df_noise_sampled, df_inj_tag, 'Inj_Prob', 'Roc', 'CNN output',
             thr=0.00001, color=colors[0], linewidth=5.5, ax=ax_roc)
        RoC(df_noise_sampled, df_inj_tag, 'max_snr', 'Roc', r'max $\rho$',
             thr=0.005, color=colors[1], linewidth=3.75, ax=ax_roc)
        if 'max_rwsnr' in df_inj_tag.columns:
            RoC(df_noise_sampled, df_inj_tag, 'max_rwsnr', 'Roc', r'max $\rho_\mathrm{rw}$',
                 thr=0.0005, color=colors[2], linewidth=2.5, ax=ax_roc)

        ax_roc.set_title(tag_title)
        ax_roc.set_xlabel('FAP')
        ax_roc.set_ylabel('TAP')
        ax_roc.grid(True)
        ax_roc.legend()
        ax_roc.set_yscale('log')
        ax_roc.set_xscale('log')
        ax_roc.set_ylim([0.1, 1.1])

        # --- bottom row: scatter ---
        if 'distance' in df_inj_tag.columns:
            sc = ax_scatter.scatter(df_inj_tag['distance'], df_inj_tag['optimal_snr'],
                                      c=df_inj_tag['Inj_Prob'], cmap='copper', s=1.2)
            ax_scatter.set_xlabel('Distance [Mpc]')
            ax_scatter.set_ylabel(r'$\rho_\mathrm{opt}$')
            ax_scatter.set_xscale('log')
            cbar = fig.colorbar(sc, ax=ax_scatter)
            cbar.set_label('Injection Probability')

    plt.tight_layout()
    plt.savefig(path_data + 'RocScatter_AllTags.png', dpi=300)
    plt.close(fig)