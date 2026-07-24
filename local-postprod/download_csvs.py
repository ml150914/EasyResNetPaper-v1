#!/usr/bin/env python3
"""Download all CSV files listed in an Apache/lighttpd-style directory index."""

import argparse
import os
from urllib.parse import unquote, urljoin

import requests
from bs4 import BeautifulSoup

DEFAULT_URL = "http://et-vd.ijclab.in2p3.fr/~lorenzo-mobilia/EasyResNetPaper-v1/S4/results_70k/"

# Original filename -> cleaner base name (without extension or dataset suffix)
CLEAN_NAME_OVERRIDES = {
    "Roc_RoC_CNN output.csv": "RoC_CNN output",
    "Roc_RoC_max $\\rho$.csv": "RoC_max_rho",
    "test_scores.csv": "test_scores",
}


def clean_base_name(filename: str) -> str:
    if filename in CLEAN_NAME_OVERRIDES:
        return CLEAN_NAME_OVERRIDES[filename]
    return os.path.splitext(filename)[0]


def download_csvs(index_url: str, out_dir: str,save_as: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    resp = requests.get(index_url, timeout=30)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    csv_links = [
        a["href"] for a in soup.find_all("a", href=True) if a["href"].lower().endswith(".csv")
    ]

    if not csv_links:
        print("No CSV files found at", index_url)
        return

    for href in csv_links:
        file_url = urljoin(index_url, href)
        filename = unquote(os.path.basename(href))
        ext = os.path.splitext(filename)[1]
        clean_base = clean_base_name(filename)
        saved_name = f"{clean_base}_{save_as}{ext}" if save_as else filename
        dest = os.path.join(out_dir, saved_name)

        print(f"Downloading {filename} -> {saved_name} ...")
        with requests.get(file_url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)

    print(f"Done. Saved {len(csv_links)} file(s) to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url", nargs="?", default=DEFAULT_URL, help="Directory index URL")
    parser.add_argument("-o", "--out-dir", default="csv_downloads", help="Output directory")
    parser.add_argument("-f", "--save-as", default="S4", help="Simulation")
    args = parser.parse_args()

    download_csvs(args.url, args.out_dir, args.save_as)
