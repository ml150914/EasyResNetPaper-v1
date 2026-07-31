import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 13,
    "axes.titlesize": 15,
    "axes.titleweight": "bold",
    "axes.titlepad": 10,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.axisbelow": True,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "legend.frameon": True,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "0.8",
    "legend.fontsize": 11,
    "figure.autolayout": False,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# Fixed, colorblind-safe (Okabe-Ito) colors assigned by *meaning*, reused
# consistently across every figure so the same color always denotes the
# same population or statistic.
COLORS = {
    'noise': '#0072B2',        # blue
    'injections': '#D55E00',   # vermillion
    'cnn_output': '#009E73',   # green   -- Inj_Prob (CNN output probability)
    'max_snr': '#E69F00',      # orange
    'max_rwsnr': '#CC79A7',    # purple
}

# Human-readable axis labels for the raw column names.
PRETTY_LABELS = {
    'Inj_Prob': 'CNN output',
    'max_snr': r'$\rho_\mathrm{max}$',
    'max_rwsnr': r'$\rho_\mathrm{rw,max}$',
}


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


def _grid(ax, minor=True):
    """Recessive major/minor grid, consistent across every plot."""
    ax.grid(True, which='major', alpha=0.3, linestyle='--')
    if minor:
        ax.grid(True, which='minor', alpha=0.12, linestyle=':')


def plot_CDF(df1, df2, column, var, label1='Noise', label2='Injections',
             save=True, thr=0.0001, colors=('b', 'r'), linewidth=1.5, ax=None,
             end_marker='*', end_marker_size=140):
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
    :param ax: axes to draw on (defaults to the current axes)
    :param end_marker: marker style for the last non-zero point of each curve
        (on a log y-axis, CDF = 0 can't be plotted, so the curve appears to stop)
    :param end_marker_size: marker size for the end marker
    :return: DataFrame with columns 'threshold', f'CDF_{label1}', f'CDF_{label2}'
    """
    if column not in df1.columns:
        raise ValueError(f"Column '{column}' not found in the DataFrame.")
    if ax is None:
        ax = plt.gca()

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

    # Mask exact zeros so the line actually stops at the last non-zero point
    # instead of matplotlib clipping it down to the log-axis floor.
    cdf1_plot = np.where(cdf1 > 0, cdf1, np.nan)
    cdf2_plot = np.where(cdf2 > 0, cdf2, np.nan)
    ax.plot(thresholds, cdf1_plot, label=f'{label1}', c=colors[0], lw=linewidth)
    ax.plot(thresholds, cdf2_plot, label=f'{label2}', c=colors[1], lw=linewidth)

    if end_marker:
        for cdf, color in ((cdf1, colors[0]), (cdf2, colors[1])):
            nonzero = np.flatnonzero(cdf > 0)
            if nonzero.size:
                idx = nonzero[-1]
                ax.scatter(thresholds[idx], cdf[idx], marker=end_marker, s=end_marker_size,
                           c=color, zorder=5, edgecolors='black', linewidths=0.5)

    return cdf_df


def finalize_cdf_axis(ax, title, xlabel, show_end_marker_note=True):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r'$\mathrm{CDF}_t(\hat r)$')
    ax.set_yscale('log')
    _grid(ax)

    handles, labels = ax.get_legend_handles_labels()
    if show_end_marker_note:
        handles.append(Line2D([0], [0], marker='*', color='none', markerfacecolor='0.3',
                               markeredgecolor='black', markersize=11, linestyle='None'))
    ax.legend(handles, labels, loc='best')


def save_cdf(label, title, column):
    pretty = PRETTY_LABELS.get(column, column)
    finalize_cdf_axis(plt.gca(), f'CDF -- {title}', pretty)
    plt.tight_layout()
    plt.savefig(path_data + f'CDF_{label}_{column}.png')
    plt.close()


def RoC(df1, df2, column, var, label, save=True, thr=0.0001, color='b', linewidth=1.5, ax=None,
        end_marker='*', end_marker_size=140):
    if ax is None:
        ax = plt.gca()
    if column not in df1.columns:
        raise ValueError(f"Column '{column}' not found in the DataFrame.")

    data1 = df1[column].dropna()
    data2 = df2[column].dropna()
    thresholds = np.arange(min(data1), max(data1), thr)

    count_above_threshold1 = CDF(data1, thresholds)
    count_above_threshold2 = CDF(data2, thresholds)

    cumulative_df = pd.DataFrame({
        'threshold': thresholds,
        'count_above_threshold1': count_above_threshold1,
        'count_above_threshold2': count_above_threshold2
    })

    if save:
        cumulative_df.to_csv(path_data + f'/{var}_RoC_{column}.csv', index=False)

    # Mask exact zeros so the line actually stops at the last non-zero point
    # instead of matplotlib clipping it down to the log-axis floor.
    fap_plot = np.where(count_above_threshold1 > 0, count_above_threshold1, np.nan)
    tap_plot = np.where(count_above_threshold2 > 0, count_above_threshold2, np.nan)
    ax.plot(fap_plot, tap_plot, label=label, c=color, lw=linewidth)

    if end_marker:
        # threshold increases -> FAP and TAP both decrease towards 0, but not
        # necessarily at the same threshold; the marker must sit on the last
        # point where *both* coordinates are still positive, i.e. the actual
        # last point the (log-log) line renders.
        nonzero_both = cumulative_df[(cumulative_df['count_above_threshold1'] > 0) &
                                      (cumulative_df['count_above_threshold2'] > 0)]
        if not nonzero_both.empty:
            end_point = nonzero_both.iloc[-1]
            ax.scatter(end_point['count_above_threshold1'], end_point['count_above_threshold2'],
                       marker=end_marker, s=end_marker_size, c=color, zorder=5,
                       edgecolors='black', linewidths=0.5)

    return cumulative_df


def finalize_roc_axis(ax, title, ylim=None, show_chance_line=False, show_end_marker_note=True):
    """Applies consistent labels, scales, grid, a chance-level reference line,
    and a legend entry explaining the star end-marker to a ROC axis."""
    ax.set_title(title)
    ax.set_xlabel('False Alarm Probability (FAP)')
    ax.set_ylabel('True Alarm Probability (TAP)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    if ylim is not None:
        ax.set_ylim(ylim)
    _grid(ax)

    if show_chance_line:
        xlim, ylim_now = ax.get_xlim(), ax.get_ylim()
        lo = max(xlim[0], ylim_now[0])
        hi = min(xlim[1], ylim_now[1])
        if lo < hi:
            ax.plot([lo, hi], [lo, hi], color='0.6', lw=1, ls=':', zorder=1,
                     label='Chance level (TAP = FAP)')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim_now)

    handles, labels = ax.get_legend_handles_labels()
    if show_end_marker_note:
        handles.append(Line2D([0], [0], marker='*', color='none', markerfacecolor='0.3',
                               markeredgecolor='black', markersize=11, linestyle='None'))
    ax.legend(handles, labels, loc='lower right')


def scatter_distance(df, title, ax, fig, colorbar=True, cmap='copper', point_size=4, alpha=0.8, cax=None):
    """
    :param cax: if given, draw the colorbar into this axes instead of
        stealing space from `ax` via fig.colorbar(sc, ax=ax). Use this when
        `ax` must keep the exact same width as sibling axes that have no
        colorbar of their own (e.g. a stacked ROC-over-scatter layout).
    """
    sc = ax.scatter(df['distance'], df['optimal_snr'], c=df['Inj_Prob'], cmap=cmap,
                     s=point_size, alpha=alpha, linewidths=0, vmin=0, vmax=1)
    ax.set_xlabel('Distance [Mpc]')
    ax.set_ylabel(r'$\rho_\mathrm{opt}$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    # Suppress minor-tick labels: on a narrow distance range they crowd and
    # overlap, especially in the multi-panel tag grid.
    ax.xaxis.set_minor_formatter(NullFormatter())
    _grid(ax)
    if title:
        ax.set_title(title)
    if colorbar:
        cbar = fig.colorbar(sc, cax=cax) if cax is not None else fig.colorbar(sc, ax=ax)
        cbar.set_label('CNN output')
    return sc


def save_roc(label, title):
    finalize_roc_axis(plt.gca(), f'ROC -- {title}')
    plt.tight_layout()
    plt.savefig(path_data + f'Roc_Superimposed_{label}.png')
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
    # A 2x2 GridSpec (plot column + colorbar column) instead of a plain 2x1
    # stack: GridSpec shares column widths across rows, so reserving a
    # colorbar column only in the bottom row still forces the top ROC axes
    # to match the scatter axes' width, keeping the two panels aligned.
    # constrained_layout does the spacing so tight_layout isn't needed (and
    # would fight the explicit width_ratios).
    fig = plt.figure(figsize=(7, 10), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[20, 1], height_ratios=[1, 1])
    ax_roc = fig.add_subplot(gs[0, 0])
    ax_scatter = fig.add_subplot(gs[1, 0])
    cax = fig.add_subplot(gs[1, 1])

    # --- Top row: ROC ---
    RoC(df_noise, df_inj, 'Inj_Prob', 'Roc', 'CNN output',
         thr=0.00001, color=COLORS['cnn_output'], linewidth=2.5, ax=ax_roc)
    RoC(df_noise, df_inj, 'max_snr', 'Roc', r'max $\rho$',
         thr=0.005, color=COLORS['max_snr'], linewidth=2.5, ax=ax_roc)
    if 'max_rwsnr' in df_inj.columns:
        RoC(df_noise, df_inj, 'max_rwsnr', 'Roc', r'max $\rho_\mathrm{rw}$',
             thr=0.0005, color=COLORS['max_rwsnr'], linewidth=2.5, ax=ax_roc)

    finalize_roc_axis(ax_roc, f'ROC -- {tag_title}')

    # --- Bottom row: scatter ---
    if 'distance' in df_inj.columns:
        scatter_distance(df_inj, tag_title, ax_scatter, fig, cax=cax)
    else:
        ax_scatter.set_visible(False)
        cax.set_visible(False)

    plt.savefig(path_data + f'RocScatter_{save_label}.png')
    plt.close(fig)


def plot_hist_comparison(df_inj, df_noise, column, xlabel, save_path, bins=60):
    """
    Overlaid, density-normalized histograms of `column` for injections vs.
    noise, sharing bin edges so the two shapes are directly comparable
    regardless of the (very different) sample sizes.
    """
    data_inj = df_inj[column].dropna()
    data_noise = df_noise[column].dropna()
    lo = min(data_inj.min(), data_noise.min())
    hi = max(data_inj.max(), data_noise.max())
    bin_edges = np.linspace(lo, hi, bins + 1)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(data_inj, bins=bin_edges, density=True, color=COLORS['injections'],
            alpha=0.55, label=f'Injections')
    ax.hist(data_noise, bins=bin_edges, density=True, color=COLORS['noise'],
            alpha=0.55, label=f'Noise')
    ax.set_yscale('log')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Probability density')
    ax.set_title(f'{xlabel} distribution')
    _grid(ax)
    ax.legend(loc='best')
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def cumulative_stat(df1, df2, stat, label, save):
    max_stat = np.arange(min(df2[stat]), max(df1[stat]), 0.01)

    cumulative_stat_df1 = CDF(df1[stat], max_stat)
    cumulative_stat_df2 = CDF(df2[stat], max_stat)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.step(max_stat, cumulative_stat_df1, label=f'Injections (n={len(df1)})', c=COLORS['injections'], lw=1.75)
    ax.step(max_stat, cumulative_stat_df2, label=f'Noise (n={len(df2)})', c=COLORS['noise'], lw=1.75)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel(PRETTY_LABELS.get(label, label))
    ax.set_title(f'Cumulative distribution -- {PRETTY_LABELS.get(label, label)}')
    _grid(ax)
    ax.legend(loc='best')
    fig.tight_layout()
    fig.savefig(path_data + f'{save}.png')
    plt.close(fig)


simulation = 'S1'

file_inj = f'/Users/lorenzomobilia/Desktop/Lavoro/PhDUrbino/IJC_lab/EasyResNetPaper-v1-1/local-postprod/csv_downloads/dfInj_test_dataset_{simulation}.csv'
file_noise = f'/Users/lorenzomobilia/Desktop/Lavoro/PhDUrbino/IJC_lab/EasyResNetPaper-v1-1/local-postprod/csv_downloads/dfNoise_test_dataset_{simulation}.csv'
path_data = f'/Users/lorenzomobilia/Desktop/Lavoro/PhDUrbino/IJC_lab/EasyResNetPaper-v1-1/local-postprod/plots/{simulation}/'

df_inj = pd.read_csv(file_inj)
df_noise = pd.read_csv(file_noise)
df_inj.dropna(how='all', axis=1, inplace=True)
df_noise.dropna(how='all', axis=1, inplace=True)

plt.figure(figsize=(7, 5.5))
plot_CDF(df_noise, df_inj, 'Inj_Prob', simulation, label1='Noise', label2='Injections',
          thr=0.00001, colors=(COLORS['noise'], COLORS['injections']), linewidth=2.5)
save_cdf(simulation, simulation, 'Inj_Prob')

plt.figure(figsize=(7, 5.5))
plot_CDF(df_noise, df_inj, 'max_snr', simulation, label1='Noise', label2='Injections',
          thr=0.001, colors=(COLORS['noise'], COLORS['injections']), linewidth=2.5)
save_cdf(simulation, simulation, 'Max_snr')

plt.figure(figsize=(7, 5.5))
RoC(df_noise, df_inj, 'Inj_Prob', 'Roc', 'CNN output', thr=0.00001, color=COLORS['cnn_output'], linewidth=2.5)
RoC(df_noise, df_inj, 'max_snr', 'Roc', r'max $\rho$', thr=0.005, color=COLORS['max_snr'], linewidth=2.5)

if 'max_rwsnr' in df_inj.columns:
    RoC(df_noise, df_inj, 'max_rwsnr', 'Roc', r'max $\rho_\mathrm{rw}$', thr=0.0005, color=COLORS['max_rwsnr'], linewidth=2.5)
else:
    print('max_rwsnr not present')

save_roc(simulation, simulation)

if 'max_rwsnr' in df_inj.columns: 
    plot_CDF(df_noise, df_inj, 'max_rwsnr', simulation, label1='Noise', label2='Injections',
          thr=0.001, colors=(COLORS['noise'], COLORS['injections']), linewidth=2.5)
    save_cdf(simulation, simulation, 'max_rwsnr')

plot_hist_comparison(df_inj, df_noise, 'max_snr', PRETTY_LABELS['max_snr'], path_data + 'max_snr.png')
plot_hist_comparison(df_inj, df_noise, 'Inj_Prob', PRETTY_LABELS['Inj_Prob'], path_data + 'inj_prob.png')


if 'distance' in df_inj.columns:
    fig, ax = plt.subplots(figsize=(7, 5.5))
    scatter_distance(df_inj, simulation, ax, fig)
    fig.tight_layout()
    fig.savefig(path_data + f'Distance_opt_snr_{simulation}.png')
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
    fig, axes = plt.subplots(2, n_cols, figsize=(5 * n_cols, 12))
    # axes[0, j] = ROC for tag j (top row)
    # axes[1, j] = scatter for tag j (bottom row)

    last_scatter = None

    n = len(tag_configs)
    fig = plt.figure(figsize=(4.2 * n, 9), layout='constrained')
    gs = fig.add_gridspec(2, n + 1, width_ratios=[1] * n + [0.04])
    axes = np.array([[fig.add_subplot(gs[i, j]) for j in range(n)] for i in range(2)])
    cax = fig.add_subplot(gs[1, -1])
    for j, (df_inj_tag, n_detected, tag_title) in enumerate(tag_configs):
        ax_roc = axes[0, j]
        ax_scatter = axes[1, j]

        df_noise_sampled = df_noise.sample(n=n_detected)

        # --- top row: ROC ---
        RoC(df_noise_sampled, df_inj_tag, 'Inj_Prob', 'Roc', 'CNN output',
             thr=0.00001, color=COLORS['cnn_output'], linewidth=2.5, ax=ax_roc)
        RoC(df_noise_sampled, df_inj_tag, 'max_snr', 'Roc', r'max $\rho$',
             thr=0.005, color=COLORS['max_snr'], linewidth=2.5, ax=ax_roc)
        if 'max_rwsnr' in df_inj_tag.columns:
            RoC(df_noise_sampled, df_inj_tag, 'max_rwsnr', 'Roc', r'max $\rho_\mathrm{rw}$',
                 thr=0.0005, color=COLORS['max_rwsnr'], linewidth=2.5, ax=ax_roc)

        finalize_roc_axis(ax_roc, f'{tag_title}')

        # --- bottom row: scatter ---
        if 'distance' in df_inj_tag.columns:
            last_scatter = scatter_distance(df_inj_tag, None, ax_scatter, fig, colorbar=False)

    # One shared colorbar for the whole bottom row: every panel uses the same
    # vmin/vmax (fixed in scatter_distance), so colors are directly comparable
    # across tags and a single legend replaces five redundant ones.
    if last_scatter is not None:
        cbar = fig.colorbar(last_scatter, cax=cax)
        cbar.set_label('CNN output')

    fig.suptitle(f'ROC and distance-SNR scatter by injection type -- {simulation}', y=1.100)
    plt.savefig(path_data + 'RocScatter_AllTags.png')
    plt.close(fig)
