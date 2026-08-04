#!/usr/bin/env python3
"""
2x3 figure built purely from PNGs already sitting in gwf/:

    top row     raw TT-SNR maps (no axes) -> ticks stamped on here
    bottom row  the Q-scan PNGs, pasted as-is (they already carry their own
                axes, labels and colourbar)

No pycbc, no lalframe, no frame reading -- this only opens images.

Tick calibration for the top row is the one from add_axes_to_ttmaps.py:
XTICK_STEP = 1000 px at 2048 Hz for time, YTICK_FRACTIONS + the template bank
for chirp mass. The bottom row is left untouched, so the two rows have
independent time axes and are not expected to line up column by column.
"""

import os

import h5py
import numpy as np
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 15,
    "axes.titlesize": 17,
    "axes.titleweight": "bold",
    "axes.titlepad": 10,
    "axes.labelsize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.axisbelow": True,
    "legend.frameon": True,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "0.8",
    "legend.fontsize": 13,
    "figure.autolayout": False,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

GWF_DIR = "gwf"
OUTPUT_DIR = ("/Users/lorenzomobilia/Desktop/Lavoro/PhDUrbino/IJC_lab/"
              "EasyResNetPaper-v1-1/local-postprod/plots")
OUT_PATH = f"{OUTPUT_DIR}/combined_TTmap_qscan.png"
BANK_PATH = ("/Users/lorenzomobilia/Desktop/Lavoro/PhDUrbino/IJC_lab/"
             "EasyResNetPaper-v1-1/bank_creation/template_bank_2_300_30Hz.h5")

# TT-map calibration
SAMPLING_RATE = 2048
XTICK_STEP = 1000
TT_NCOLS = 8192
YTICK_FRACTIONS = np.array([0.0798, 0.2635, 0.4471, 0.6307, 0.8142, 0.9978])

PANEL_WIDTH = 4.5   # inches per column

# (title, TT-map PNG, Q-scan PNG) -- both relative to GWF_DIR
PANELS = [
    ("Gaussian Noise",     "TT_map_NOISE_0.png",      "noise_0_qscan.png"),
    ("Injection",          "TT_map_SNR_0.png",        "injection_0_qscan.png"),
    ("Injection + Glitch", "TT_map_SNR_GLITCH_0.png", "injection_with_glitch_0_qscan.png"),
]

TICK_PIXELS = np.arange(0, TT_NCOLS, XTICK_STEP)
TICK_LABELS = [f"{t:.2g}" for t in TICK_PIXELS / SAMPLING_RATE]


def resolve(name):
    path = os.path.join(GWF_DIR, name)
    if not os.path.exists(path):
        available = sorted(f for f in os.listdir(GWF_DIR) if f.endswith(".png"))
        raise FileNotFoundError(
            f"{path} not found. PNGs present in {GWF_DIR}/:\n  "
            + "\n  ".join(available)
        )
    return path


def load_bank_chirp_masses(bank_path=BANK_PATH):
    """Chirp mass of every template, in the bank's native ascending order."""
    with h5py.File(bank_path, "r") as f:
        m1, m2 = f["mass1"][:], f["mass2"][:]
    return (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2


def row_to_mc(fraction, chirp_masses):
    """Fraction of image height (0 = top = highest mass) -> Mc [Msun]."""
    n = len(chirp_masses)
    i = int(round(np.clip(n - 1 - fraction * (n - 1), 0, n - 1)))
    return chirp_masses[i]


def draw_tt_map(ax, arr, chirp_masses, show_ylabel):
    """Raw TT-map image with Time [s] / Mc [Msun] ticks stamped on."""
    ax.imshow(arr, origin="upper", interpolation="antialiased", aspect="auto")

    # Rescale the nominal pixel ticks if this render isn't exactly TT_NCOLS wide.
    ax.set_xticks(TICK_PIXELS * arr.shape[1] / TT_NCOLS)
    ax.set_xticklabels(TICK_LABELS)
    ax.set_xlim(0, arr.shape[1])
    ax.set_xlabel("Time [s]")

    ax.set_yticks(np.round(YTICK_FRACTIONS * (arr.shape[0] - 1)).astype(int))
    if show_ylabel:
        ax.set_yticklabels([f"{row_to_mc(f, chirp_masses):.2g}" for f in YTICK_FRACTIONS])
        ax.set_ylabel(r"$M_\mathrm{c}$ [$M_\odot$]")
    else:
        ax.set_yticklabels([])


def paste_qscan(ax, arr):
    """Drop a finished Q-scan PNG in as-is, aspect preserved, no matplotlib axes."""
    ax.imshow(arr, origin="upper", interpolation="antialiased")
    ax.axis("off")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    chirp_masses = load_bank_chirp_masses()

    tt_imgs = [np.asarray(Image.open(resolve(tt)).convert("RGB")) for _, tt, _ in PANELS]
    q_imgs = [np.asarray(Image.open(resolve(q)).convert("RGB")) for _, _, q in PANELS]

    # Row heights follow the images' own aspect ratios, so the Q-scan PNGs --
    # which contain baked-in text and a colourbar -- are never stretched.
    tt_ar = tt_imgs[0].shape[0] / tt_imgs[0].shape[1]
    q_ar = q_imgs[0].shape[0] / q_imgs[0].shape[1]

    fig, axes = plt.subplots(
        2, 3,
        figsize=(3 * PANEL_WIDTH, PANEL_WIDTH * (tt_ar + q_ar) + 0.6),
        gridspec_kw={"height_ratios": [tt_ar, q_ar]},
        layout="constrained",
    )

    for col, (title, _, _) in enumerate(PANELS):
        draw_tt_map(axes[0, col], tt_imgs[col], chirp_masses, show_ylabel=(col == 0))
        axes[0, col].set_title(title)
        paste_qscan(axes[1, col], q_imgs[col])

    fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] wrote {OUT_PATH}")


if __name__ == "__main__":
    main()