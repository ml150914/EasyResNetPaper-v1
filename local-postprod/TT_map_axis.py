#!/usr/bin/env python3
"""
Adds Time / chirp-mass axis ticks and labels to the plain (axis-less) TT-SNR
MAP PNGs used in the paper, producing the publication figures:
Eccentricity_axis.png, HM_axis.png, Precessing_axis.png, Superimposed_axis.png,
Extreme_Spin_axis.png.

Ported from NicePlots.ipynb (cells 16-17, in
/Users/lorenzomobilia/Desktop/Lavoro/PhDUrbino/ResNetPoster/). It performs no
matched filtering and needs no pycbc: it just opens the already-rendered raw
TT-map image and stamps matplotlib ticks on it.

Axis calibration, checked against ResNet_Article_reviewed.pdf (Sec. III):
- X (time): the paper states the TT-SNR map spans a 4s window at 2048 Hz,
  i.e. M = 8192 time samples. `xtick_step=1000` px / `SAMPLING_RATE=2048`
  reproduces exactly the labels on the existing Eccentricity_axis.png
  (0, 0.49, 0.98, 1.5, 2, 2.4, 2.9, 3.4, 3.9) -- verified correct.
- Y (chirp mass): the paper states the bank has 5,448 templates spanning
  Mc in [0.87, 130.6] Msun, matching bank_creation/template_bank_2_300_30Hz.h5
  exactly (checked directly: 5448 templates, Mc in [0.8706, 130.58]). Y-tick
  labels and any Mc-based crop (see mc_to_row/time_lim/mc_lim below) are
  computed directly from that bank file at runtime, rather than hardcoded,
  using the same `row = N-1-i` convention as TT_map_batch.py (mc sorted
  ascending in i). This needs h5py (present in the `gw_env` conda env used to
  run this script).
  Note the bank is extremely sparse above Mc~4 (only 6 of 5448 templates
  exceed Mc=30), so "nice" round tick values (e.g. 10, 30, 100) all collapse
  into the unreadable top few pixel rows -- ticks are kept at fixed,
  evenly-spaced rows instead, each labelled with the true Mc found there.
  YTICK_FRACTIONS below are expressed as a fraction of image height (0 = top
  = highest-mass template) so they scale correctly regardless of the actual
  source image's row count (the raw example maps we found range from 5436 to
  5448 rows depending on the specific run).
"""

import os
import urllib.request

import h5py
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
})

FIGURES_DIR = "/Users/lorenzomobilia/Desktop/Lavoro/PhDUrbino/IJC_lab/ResNet-Article/easyresnetpaper/Figures"
OUTPUT_DIR = "/Users/lorenzomobilia/Desktop/Lavoro/PhDUrbino/IJC_lab/EasyResNetPaper-v1-1/local-postprod/plots"
BANK_PATH = "/Users/lorenzomobilia/Desktop/Lavoro/PhDUrbino/IJC_lab/EasyResNetPaper-v1-1/bank_creation/template_bank_2_300_30Hz.h5"

SAMPLING_RATE = 2048  # Hz, used to convert pixel column <-> time [s]
XTICK_STEP = 1000  # pixels
YTICK_FRACTIONS = np.array([0.0798, 0.2635, 0.4471, 0.6307, 0.8142, 0.9978])

# Zoom around the interference/crossing region where the two overlapping BNS
# chirps meet, in human-friendly units (Time in seconds, Mc in Msun) -- in the
# style of the reference Old_ttmap/Superimposed_axis_zoom.png. Re-tuned by eye
# on the current FAMILIES["Superimposed"] source image (the crossing point
# moves if that source is swapped for a different injection).
SUPERIMPOSED_ZOOM_TIME = (1.8, 2.3)
SUPERIMPOSED_ZOOM_MC = (1.12, 1.32)

# family -> (source URL on the public results mirror, cached local filename,
# axis-labelled output filename). The URLs point at full-resolution
# (8192x5436) raw TT-map examples -- verified to match the x-axis calibration
# above -- one representative injection per S4 simulation family.
FAMILIES = {
    "Eccentricity": (
        "http://et-vd.ijclab.in2p3.fr/~lorenzo-mobilia/EasyResNetPaper-v1/S4/examples/Eccentricity/TT_map_SNR_19526.png",
        "Eccentricity.png", "Eccentricity_axis.png",
    ),
    "HM": (
        "http://et-vd.ijclab.in2p3.fr/~lorenzo-mobilia/EasyResNetPaper-v1/S4/examples/HoM/TT_map_SNR_321.png",
        "HM.png", "HM_axis.png",
    ),
    "Precessing": (
        "http://et-vd.ijclab.in2p3.fr/~lorenzo-mobilia/EasyResNetPaper-v1/S4/examples/Precessing/TT_map_SNR_1.png",
        "Precessing.png", "Precessing_axis.png",
    ),
    "Superimposed": (
        "http://et-vd.ijclab.in2p3.fr/~lorenzo-mobilia/EasyResNetPaper-v1/S4/examples/Superimposed/TT_map_SNR_98.png",
        "Superimposed.png", "Superimposed_axis.png",
    ),
    "Extreme_Spin": (
        "http://et-vd.ijclab.in2p3.fr/~lorenzo-mobilia/EasyResNetPaper-v1/S4/examples/ExtremeSpin/TT_map_SNR_44.png",
        "Extreme_spin.png", "Extreme_Spin_axis.png",
    ),
}

# Thesis Chapter 5 side-by-side panel (Gaussian Noise / Injection / Injection +
# Glitch), ported from NicePlots.ipynb's "create side by side example picture"
# cells. These raw full-resolution (8192x5436) TT-maps already exist locally,
# no download needed.
THESIS_TTMAP_DIR = "/Users/lorenzomobilia/Desktop/Lavoro/PhDUrbino/Tesi/tikz-pic"
THESIS_PANELS = [
    ("Gaussian Noise", f"{THESIS_TTMAP_DIR}/TTmap_GN_0.png"),
    ("Injection", f"{THESIS_TTMAP_DIR}/TTmap_BNS_0.png"),
    ("Injection + Glitch", f"{THESIS_TTMAP_DIR}/TTmap_BNS_GLITCH_0.png"),
]
THESIS_COMBINED_OUTPUT = f"{OUTPUT_DIR}/combined_TTmaps.png"


def fetch_source_image(url, cache_path):
    """Download the raw TT-map PNG if not already cached locally."""
    if not os.path.exists(cache_path):
        print(f"Downloading {url} -> {cache_path}")
        urllib.request.urlretrieve(url, cache_path)
    return Image.open(cache_path)


def load_bank_chirp_masses(bank_path=BANK_PATH):
    """Chirp mass of every template in the bank, ascending (the bank's native
    ordering -- see bank_creation/template_bank_2_300_30Hz.h5)."""
    with h5py.File(bank_path, "r") as f:
        m1 = f["mass1"][:]
        m2 = f["mass2"][:]
    return (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2


def time_to_pixel(t):
    """Time [s] -> pixel column."""
    return t * SAMPLING_RATE


def mc_to_row(mc, chirp_masses, img_height):
    """Chirp mass [Msun] -> pixel row in an image of the given height.

    Row 0 = top = highest-mass template, per TT_map_batch.py's
    `row = num_templates - 1 - i` convention. Scaled by
    img_height / len(chirp_masses) so it stays correct even if the loaded
    image's row count doesn't exactly match the bank (raw examples we found
    range from 5436 to 5448 rows depending on the specific run).
    """
    n_bank = len(chirp_masses)
    i = int(np.clip(np.searchsorted(chirp_masses, mc), 0, n_bank - 1))
    row_in_bank = n_bank - 1 - i
    return row_in_bank * (img_height - 1) / (n_bank - 1)


def row_to_mc(fraction, chirp_masses):
    """Inverse of mc_to_row: fraction of image height (0 = top) -> Mc [Msun]."""
    n_bank = len(chirp_masses)
    i = int(round(np.clip(n_bank - 1 - fraction * (n_bank - 1), 0, n_bank - 1)))
    return chirp_masses[i]


def draw_ticked_image(ax, img, xtick_step, ytick_fractions, chirp_masses,
                       time_lim=None, mc_lim=None, ylabel=True):
    """Draw a raw TT-map image onto `ax` with Time [s] / M_c [Msun] ticks.

    time_lim=(t0, t1) [s] and mc_lim=(mc0, mc1) [Msun] crop the displayed
    view to a region of interest -- e.g. around the merger -- while keeping
    the same tick calibration as the full image, so labels stay physically
    correct. Ticks that fall outside the cropped view are simply not shown.
    """
    # Force RGB so matplotlib treats it as an image, not scalar data -> no colormap mapping
    img_rgb = img.convert("RGB")
    arr = np.asarray(img_rgb)

    ax.imshow(arr, origin="upper", interpolation="antialiased")
    ax.set_aspect("equal")

    xticks = np.arange(0, arr.shape[1], xtick_step)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{x / SAMPLING_RATE:.2g}" for x in xticks])

    yticks = np.round(ytick_fractions * (arr.shape[0] - 1)).astype(int)
    ylabels = [f"{row_to_mc(f, chirp_masses):.2g}" for f in ytick_fractions]
    ax.set_yticks(yticks)
    if ylabel:
        ax.set_yticklabels(ylabels)
        ax.set_ylabel(r"$M_\mathrm{c}$ [$M_\odot$]")
    else:
        ax.set_yticklabels([])

    ax.set_xlabel("Time [s]")

    if time_lim is not None:
        ax.set_xlim(time_to_pixel(time_lim[0]), time_to_pixel(time_lim[1]))
    if mc_lim is not None:
        # mc_lim[0] (lower mass) -> bottom of the view (larger row);
        # mc_lim[1] (higher mass) -> top of the view (smaller row).
        ax.set_ylim(
            mc_to_row(mc_lim[0], chirp_masses, arr.shape[0]),
            mc_to_row(mc_lim[1], chirp_masses, arr.shape[0]),
        )


def apply_ticks(img, xtick_step, ytick_fractions, chirp_masses, path_for_saving,
                 time_lim=None, mc_lim=None):
    """Stamp Time [s] / M_c [Msun] axis ticks onto a raw TT-map image and save it."""
    fig, ax = plt.subplots()
    draw_ticked_image(ax, img, xtick_step, ytick_fractions, chirp_masses,
                       time_lim=time_lim, mc_lim=mc_lim)
    fig.savefig(path_for_saving, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_combined_panel(panels, xtick_step, ytick_fractions, chirp_masses, path_for_saving):
    """Combine several (title, raw TT-map path) pairs into one side-by-side,
    axis-labelled figure -- ported from NicePlots.ipynb's "create side by side
    example picture" cells (Gaussian Noise / Injection / Injection + Glitch).
    Only the leftmost panel keeps the M_c y-axis label/ticks, matching the
    reference combined_TTmaps.png style.
    """
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 5), dpi=300)
    for i, (ax, (title, path)) in enumerate(zip(axes, panels)):
        img = Image.open(path)
        draw_ticked_image(ax, img, xtick_step, ytick_fractions, chirp_masses, ylabel=(i == 0))
        ax.set_title(title)

    fig.tight_layout()
    fig.savefig(path_for_saving, dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    chirp_masses = load_bank_chirp_masses()

    for name, (url, src_name, out_name) in FAMILIES.items():
        src_path = f"{FIGURES_DIR}/{src_name}"
        out_path = f"{OUTPUT_DIR}/{out_name}"
        img = fetch_source_image(url, src_path)
        apply_ticks(img, XTICK_STEP, YTICK_FRACTIONS, chirp_masses, out_path)
        print(f"[ok] {name}: {src_path} -> {out_path}")

        if name == "Superimposed":
            zoom_out_path = f"{OUTPUT_DIR}/Superimposed_axis_zoom.png"
            apply_ticks(
                img, XTICK_STEP, YTICK_FRACTIONS, chirp_masses, zoom_out_path,
                time_lim=SUPERIMPOSED_ZOOM_TIME, mc_lim=SUPERIMPOSED_ZOOM_MC,
            )
            print(f"[ok] {name} (zoom): {src_path} -> {zoom_out_path}")

    save_combined_panel(THESIS_PANELS, XTICK_STEP, YTICK_FRACTIONS, chirp_masses, THESIS_COMBINED_OUTPUT)
    print(f"[ok] Thesis combined panel -> {THESIS_COMBINED_OUTPUT}")


if __name__ == "__main__":
    main()
