import os, sys, time, csv, shutil, argparse, configparser
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from paper_net import paper_net, small_se_net, wide_net
from read_data import create_tf

MODEL_MAP = {"paper_net": paper_net, "small_se_net": small_se_net, "wide_net": wide_net}


def read_png_metadata(path):
    with Image.open(path) as img:
        return {k: img.info[k] for k in img.info.keys()}

def build_args():
    p = argparse.ArgumentParser(description="Run the CNN training and test")
    p.add_argument("--config", default="config.ini",
                   help="Path to config.ini file")
    cli_args = p.parse_args()

    config = configparser.ConfigParser()
    if not os.path.exists(cli_args.config):
        raise FileNotFoundError(f"Config file not found: {cli_args.config}")
    config.read(cli_args.config)

    cfg = config["DEFAULT"]

    # Build a namespace to keep the same attribute-style access (args.xxx)
    args = argparse.Namespace(
        data_input_dir=cfg.get("data_input_dir"),
        data_public_out=cfg.get("data_public_out"),
        data_out=cfg.get("data_out"),
        model=cfg.get("model"),
        epochs=cfg.getint("epochs", fallback=20),
        batch_size=cfg.getint("batch_size", fallback=64),
        save_images=cfg.getboolean("save_images", fallback=False),
    )

    if args.model not in MODEL_MAP:
        raise ValueError(f"Invalid model '{args.model}'. Choices: {list(MODEL_MAP)}")

    return args

def main():
    args = build_args()
    for path, name in [(args.data_input_dir, "data_input_dir"),
                       (args.data_public_out, "data_public_out"),
                       (args.data_out, "data_out")]:
        if not os.path.exists(path):
            sys.exit(f"ERROR: Directory '{path}' ({name}) does not exist.")

    img_height, img_width = 256, 512

    train_ds, val_ds, test_ds, file_path_test = create_tf(
        args.data_input_dir, img_width, img_height, args.batch_size)

    # sanity check
    for images, _ in val_ds.take(1):
        plt.figure(figsize=(9, 9))
        for i in range(min(9, images.shape[0])):
            plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().squeeze(), cmap="gray")
            plt.axis("off")
        plt.savefig(os.path.join(args.data_public_out, "sanity_check_images.png"))
        plt.close()

    # Load the model + [0,1] normalization
    base = MODEL_MAP[args.model](shape = (img_height, img_width, 1), classes = 2)
    inp = keras.Input(shape = (img_height, img_width, 1))
    out = base(layers.Rescaling(1.0 /255)(inp))
    model = keras.Model(inp, out)

    # training
    steps_per_epoch = int(train_ds.cardinality().numpy())
    if steps_per_epoch <= 0:
        steps_per_epoch = 1000

    total_steps = steps_per_epoch * args.epochs
    initial_lr = 1e-3
    lr_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate = initial_lr,
        decay_steps = total_steps,
        alpha = 1e-5 / initial_lr)
    
    model.compile(
        optimizer = keras.optimizers.Adam(learning_rate = lr_schedule),
        loss = keras.losses.SparseCategoricalCrossentropy(from_logits = False),
        metrics = ["accuracy"])
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience = 20,
                                      min_delta = 1e-2,
                                      restore_best_weights = True)
    ]
    history = model.fit(train_ds, validation_data = val_ds,
                        epochs = args.epochs,
                        callbacks = callbacks)

    model.save(os.path.join(args.data_out, f"{args.model}.keras"))

    # training curve metrics
    for metric, fname, title in [
            (("accuracy", "val_accuracy"), "Train_val_Accuracy.png", "Accuracy"),
            (("loss", "val_loss"), "Train_val_loss.png", "Loss")]:
        plt.figure()
        plt.plot(history.history[metric[0]], label=f"Train {title}")
        plt.plot(history.history[metric[1]], label=f"Validation {title}")
        plt.legend(); plt.title(f"Training and Validation {title}")
        plt.savefig(os.path.join(args.data_public_out, fname))
        plt.close()

    print('saving the metrics output')
    hist_df = pd.DataFrame(history.history)
    hist_df.index.name = 'epoch'
    hist_df.to_csv(os.path.join(args.data_public_out, 'training_history.csv'))

    # predict on the test dataset
    t0 = time.time()
    probs = model.predict(test_ds, verbose=0)
    elapsed_ms = (time.time() - t0) * 1000
    print(f"Average inference time per image: {elapsed_ms / len(probs):.2f} ms")

    # Save the results
    csv_path = os.path.join(args.data_out, "test_scores.csv")

    # inj = col 0, noise = col 1

    csv_path = os.path.join(args.data_out, "test_scores.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "Inj_Prob", "Noise_Prob", "Predicted_label",
                    "Label", "optimal_snr", "max_snr", "max_rwsnr", "chisq", "distance", "activator"])
        for i, p in enumerate(probs):
            src = file_path_test[i]
            if isinstance(src, bytes):
                src = src.decode("utf-8")
            metadata = read_png_metadata(src)
            inj_p, noise_p = float(p[0]), float(p[1])
            predicted_label = "inj" if inj_p >= 0.5 else "noise"
            true_label = metadata.get("Label")
            if true_label is None:
                true_label = 'Injection'
            w.writerow([src,
                        f"{inj_p:.6f}",
                        f"{noise_p:.6f}",
                        predicted_label,
                        true_label,
                        metadata.get("optimal_snr", ""),
                        metadata.get("max_snr", ""),
                        metadata.get("max_rwsnr", ""),
                        metadata.get("chisq", ""),
                        metadata.get("distance", ""),
                        metadata.get("activator", "")
                        ])
 
            if args.save_images == True:
                # copy of original PNG + metadata 
                dst = os.path.join(args.data_out, f"TTMap_{i}.png")
                shutil.copyfile(src, dst)
                meta = PngInfo()
                combined = read_png_metadata(src)
                combined.update({"Inj_Prob": inj_p,
                                 "Noise_Prob": noise_p,
                                 "Predicted_label": predicted_label})
                for k, v in combined.items():
                    meta.add_text(k, str(v))
                Image.open(dst).save(dst, "png", pnginfo=meta)
 
    print(f"Scores saved to {csv_path}")

if __name__ == "__main__":
    main()
