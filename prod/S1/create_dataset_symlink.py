#!/usr/bin/env python3
"""
Split a set of injection ("inj") and noise ("noise") images into
train/validation/test folders using symlinks (no disk duplication).

Expected input layout:
    <data_dir>/inj/...      (images)
    <data_dir>/noise/...    (images)

Output layout:
    <output_dir>/train/inj/...
    <output_dir>/train/noise/...
    <output_dir>/val/inj/...
    <output_dir>/val/noise/...
    <output_dir>/test/inj/...
    <output_dir>/test/noise/...

`--dataset-size` is the TOTAL number of images across both classes
(inj + noise) that will end up in the output, split according to
train/val/test fractions. The inj/noise ratio in the output mirrors
the ratio available in the input data (capped by whatever exists).
"""

import argparse
import os
import random
import uuid

IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg')


def list_files(directory):
    """Recursively list image files under `directory`."""
    files = []
    for root, _, filenames in os.walk(directory):
        for f in filenames:
            if f.lower().endswith(IMAGE_EXTENSIONS):
                files.append(os.path.join(root, f))
    return files


def split_indices(n, train_frac, val_frac, seed):
    """Return (train_idx, val_idx, test_idx) lists partitioning range(n)."""
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    train_end = int(n * train_frac)
    val_end = train_end + int(n * val_frac)
    return idx[:train_end], idx[train_end:val_end], idx[val_end:]


def symlink_files(file_list, target_dir, label):
    """Symlink each file in file_list into target_dir/label/<random_name>."""
    label_dir = os.path.join(target_dir, label)
    os.makedirs(label_dir, exist_ok=True)
    for fp in file_list:
        if not os.path.exists(fp):
            print(f"File not found: {fp}")
            continue
        ext = os.path.splitext(fp)[1] or '.png'
        unique_name = f"{uuid.uuid4().hex}{ext}"
        dest = os.path.join(label_dir, unique_name)
        os.symlink(os.path.abspath(fp), dest)


def allocate_counts(dataset_size, n_inj_available, n_noise_available):
    """Decide how many inj/noise images to draw so the total equals
    dataset_size (or the max available, if smaller), keeping the
    output ratio close to the input ratio."""
    total_available = n_inj_available + n_noise_available
    if total_available == 0:
        raise ValueError("No images found in either inj or noise directories.")

    if dataset_size > total_available:
        print(
            f"Warning: requested dataset_size={dataset_size} exceeds "
            f"available images ({total_available}). Using {total_available} instead."
        )
        dataset_size = total_available

    n_inj_target = min(n_inj_available, round(dataset_size * n_inj_available / total_available))
    n_noise_target = dataset_size - n_inj_target

    # Rebalance if one class doesn't have enough to fill its share.
    if n_noise_target > n_noise_available:
        deficit = n_noise_target - n_noise_available
        n_noise_target = n_noise_available
        n_inj_target = min(n_inj_available, n_inj_target + deficit)
    elif n_inj_target > n_inj_available:
        deficit = n_inj_target - n_inj_available
        n_inj_target = n_inj_available
        n_noise_target = min(n_noise_available, n_noise_target + deficit)

    return n_inj_target, n_noise_target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', required=True,
                         help="Directory containing the injections and noise subfolders.")
    parser.add_argument('--inj-subdir', default='injections_16_bins_correct_seed',
                         help="Name of the injections subfolder inside --data-dir.")
    parser.add_argument('--noise-subdir', default='noise_16_bins_correct_seed',
                         help="Name of the noise subfolder inside --data-dir.")
    parser.add_argument('--output-dir', required=True,
                         help="Directory where train/val/test folders will be created.")
    parser.add_argument('--dataset-size', type=int, required=True,
                         help="Total number of images (inj + noise) to include.")
    parser.add_argument('--train-frac', type=float, default=0.70)
    parser.add_argument('--val-frac', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    test_frac = 1.0 - args.train_frac - args.val_frac
    if test_frac < 0:
        raise ValueError("train-frac + val-frac must be <= 1.0")

    inj_dir = os.path.join(args.data_dir, args.inj_subdir)
    noise_dir = os.path.join(args.data_dir, args.noise_subdir)

    inj_files = list_files(inj_dir)
    noise_files = list_files(noise_dir)

    random.Random(args.seed).shuffle(inj_files)
    random.Random(args.seed + 1).shuffle(noise_files)

    n_inj_target, n_noise_target = allocate_counts(
        args.dataset_size, len(inj_files), len(noise_files)
    )

    inj_selected = inj_files[:n_inj_target]
    noise_selected = noise_files[:n_noise_target]

    inj_train_idx, inj_val_idx, inj_test_idx = split_indices(
        len(inj_selected), args.train_frac, args.val_frac, args.seed + 2
    )
    noise_train_idx, noise_val_idx, noise_test_idx = split_indices(
        len(noise_selected), args.train_frac, args.val_frac, args.seed + 3
    )

    splits = {
        'train': (
            [inj_selected[i] for i in inj_train_idx],
            [noise_selected[i] for i in noise_train_idx],
        ),
        'val': (
            [inj_selected[i] for i in inj_val_idx],
            [noise_selected[i] for i in noise_val_idx],
        ),
        'test': (
            [inj_selected[i] for i in inj_test_idx],
            [noise_selected[i] for i in noise_test_idx],
        ),
    }

    os.makedirs(args.output_dir, exist_ok=True)

    for split_name, (inj_list, noise_list) in splits.items():
        target_dir = os.path.join(args.output_dir, split_name)
        symlink_files(inj_list, target_dir, 'inj')
        symlink_files(noise_list, target_dir, 'noise')
        print(f"{split_name}: {len(inj_list)} inj + {len(noise_list)} noise "
              f"= {len(inj_list) + len(noise_list)} images")

    # Sanity check: confirm no file appears in more than one split.
    all_sets = {
        name: set(inj_list + noise_list)
        for name, (inj_list, noise_list) in splits.items()
    }
    names = list(all_sets.keys())
    overlap_found = False
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            overlap = all_sets[names[i]] & all_sets[names[j]]
            if overlap:
                overlap_found = True
                print(f"Overlap between {names[i]} and {names[j]}: {overlap}")
    if not overlap_found:
        print("No overlapping files between splits.")


if __name__ == '__main__':
    main()
