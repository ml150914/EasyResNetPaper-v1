import os, csv, glob, random, shutil, argparse

EXTS = ('.png', '.jpg', '.jpeg')

def list_images(folder):
    files = [f for f in glob.glob(os.path.join(folder, '*'))
             if f.lower().endswith(EXTS)]

    if not files:
        raise SystemExit("ERROR: NO IMAGE FOUND IN '{folder}'")

    return sorted(files)

def split_indices(n, r_train, r_val, seed, test_per_class=None):
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    if test_per_class:                      
        n_test = min(test_per_class, n)
        rest = n - n_test
        n_train = round(rest * r_train / (r_train + r_val))
        n_val = rest - n_train
    else:
        n_train = int(n * r_train)
        n_val = int(n * r_val)
    return idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]

def place(src, dst, mode):
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "move":
        shutil.move(src, dst)
    elif mode == "symlink":
        os.symlink(os.path.abspath(src), dst)


def main():
    ap = argparse.ArgumentParser(description="Split inj/noise into train/val/test")
    ap.add_argument("--inj-dir", default="injections_16_bins")
    ap.add_argument("--noise-dir", default="noise_16_bins")
    ap.add_argument("--out-dir", default="dataset")
    ap.add_argument("--test-per-class", type=int, default=None,
                    help="if set, it puts exactly N image/class in test;")
    ap.add_argument("--train", type=float, default=0.70)
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=156)
    ap.add_argument("--mode", choices=["copy", "move", "symlink"], default="copy")
    args = ap.parse_args()


    assert abs(args.train + args.val + args.test - 1.0) < 1e-6, "We are losing images!"

    # create the folders
    for sub in ("train/inj", "train/noise", "val/inj", "val/noise", "test"):
        os.makedirs(os.path.join(args.out_dir, sub), exist_ok=True)

    classes = {"inj": args.inj_dir, "noise": args.noise_dir}
    test_manifest = []
    summary = {}

    for cls, folder in classes.items():
        files = list_images(folder)
        tr, va, te = split_indices(len(files), args.train, args.val, args.seed, args.test_per_class)
        if args.test_per_class and len(files) < args.test_per_class:
            print(f"  WARNING: '{cls}' has only {len(files)} images "
                  f"(< {args.test_per_class}); train/val will be empty.")
        summary[cls] = (len(tr), len(va), len(te))

        for split, ids in (("train", tr), ("val", va)):
            for i in ids:
                src = files[i]
                place(src, os.path.join(args.out_dir, split, cls,
                                        os.path.basename(src)), args.mode)

        for i in te:
            src = files[i]
            name = os.path.basename(src)
            dst = os.path.join(args.out_dir, "test", name)
            if os.path.exists(dst):              
                name = f"{cls}_{name}"
                dst = os.path.join(args.out_dir, "test", name)
            place(src, dst, args.mode)
            test_manifest.append((name, cls))

        with open(os.path.join(args.out_dir, "test_labels.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["file", "true_label"])
            w.writerows(test_manifest)

        print(f"Output: {os.path.abspath(args.out_dir)}  (mode={args.mode}, seed={args.seed})")
        print(f"{'classe':8s} {'train':>7s} {'val':>7s} {'test':>7s}")
        for cls, (a, b, c) in summary.items():
            print(f"{cls:8s} {a:>7d} {b:>7d} {c:>7d}")
        print(f"test total (flat): {len(test_manifest)} images -> test_labels.csv")
 
 
if __name__ == "__main__":
    main()
