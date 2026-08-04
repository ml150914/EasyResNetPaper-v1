#!/usr/bin/env python
"""Download injection.gwf / noise.gwf from the index page and Q-scan each one."""

import os
from urllib.parse import unquote, urljoin

import requests
from bs4 import BeautifulSoup

import matplotlib
matplotlib.use("Agg")          # no display needed
import matplotlib.pyplot as plt

import pycbc.frame as frame
from pycbc.filter import highpass, resample_to_delta_t
from lalframe.utils import get_channels



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

LABELS = {
    "noise_0.gwf":                "Gaussian Noise",
    "injection_0.gwf":            "Injection",
    "injection_with_glitch_0.gwf": "Injection + Glitch",
}

DEFAULT_URL = ("http://et-vd.ijclab.in2p3.fr/~lorenzo-mobilia/"
               "EasyResNetPaper-v1/paper-images-combined/")
OUT_DIR = "gwf"
WANTED = {"injection_0.gwf", "noise_0.gwf", "injection_with_glitch_0.gwf"}   # set to None to grab every .gwf


def download_gwf(index_url: str, out_dir: str, wanted=None) -> dict:
    """Fetch .gwf files listed at `index_url`. Returns {filename: local path}."""
    os.makedirs(out_dir, exist_ok=True)

    resp = requests.get(index_url, timeout=30)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    hrefs = [a["href"] for a in soup.find_all("a", href=True)
             if a["href"].lower().endswith(".gwf")]

    paths = {}
    for href in hrefs:
        filename = unquote(os.path.basename(href))
        if wanted is not None and filename not in wanted:
            continue

        dest = os.path.join(out_dir, filename)
        paths[filename] = dest

        if os.path.exists(dest):
            print(f"{filename} already present, skipping")
            continue

        print(f"Downloading {filename} ...")
        with requests.get(urljoin(index_url, href), stream=True, timeout=60) as r:
            r.raise_for_status()
            tmp = dest + ".part"
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
            os.replace(tmp, dest)   # so an interrupted download isn't mistaken for a good one

    missing = (wanted or set()) - set(paths)
    if missing:
        print("WARNING: not found at the index:", ", ".join(sorted(missing)))

    print(f"{len(paths)} file(s) in {out_dir}")
    return paths


def qscan(path: str, title, channel: str = None, out_png: str = None,
          qrange=(4.0, 64.0), frange=(20.0, 512.0), vmax=None) -> None:
    channels = get_channels(path)
    if channel is None:
        channel = channels[0]
    print(f"{os.path.basename(path)}: channels={channels} -> using {channel}")

    strain = frame.read_frame(path, channels=channel)
    strain = highpass(strain, 15)
    strain = resample_to_delta_t(strain, 1.0 / 2048)
    strain = strain.crop(2, 2)

    times, freqs, power = strain.qtransform(0.001, logfsteps=200,
                                            qrange=qrange, frange=frange)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    mesh = ax.pcolormesh(times - times[0], freqs, power ** 0.5,
                         vmax=vmax, shading="auto", cmap="viridis")
    ax.set_yscale("log")
    ax.set_ylim(*frange)
    ax.set_xlabel(f"Time [s]")
    ax.set_ylabel("Frequency [Hz]")
    ax.set_title(f"{title}   Q $\\in$ {qrange}")
    fig.colorbar(mesh, ax=ax, label=r"Power")

    out_png = out_png or os.path.splitext(os.path.basename(path))[0] + "_qscan.png"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_png}")

def main():
    paths = download_gwf(DEFAULT_URL, OUT_DIR, wanted=set(LABELS))
    for filename, label in LABELS.items():
        if filename not in paths:
            continue
        stem = os.path.splitext(filename)[0]
        qscan(paths[filename], label,
              out_png=os.path.join(OUT_DIR, f"{stem}_qscan.png"))


if __name__ == "__main__":
    main()