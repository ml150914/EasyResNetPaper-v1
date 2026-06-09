import os, sys, time, csv, shutil, argparse
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


def read_pnf_metadata(path):
    with Image.open(path) as img:
        return {k: img.info[k] for k in img.info.keys()}


def build_args():
    p = argparse.ArgumentParser(description="Run the CNN training and test")
    p.add_argument("--data-input-dir", required=True)
    p.add_argument("--data-public-out", required=True)
    p.add_argument("--data-out", required=True)
    p.add_argument("--model", required=True, choices=list(MODEL_MAP))
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--save-images", action="store_true",
                   help="CSV + save test PNG +  metadata")
    return p.parse_args()

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
            keras.callbacks.EarlyStopping(monitor='val_loss', patience = 5,
                                          min_delta = 1e-3,
                                          restore_best_weights = True)
            ]
        history = model.fit(train_ds, validation_data = val_ds,
                            epochs = args.epochs,
                            callbacks = callbacks)

        model.save(os.path.join(args.data_out, f"{args.model}.keras"))

        # training curve metrics
        for metrix, fname, title in [
                (("accuracy", "val_accuracy"), "Train_val_Accuracy.png", "Accuracy"),
                (("loss", "val_loss"), "Train_val_loss.png", "Loss")]:
            plt.figure()
            plt.plot(history.history[metric[0]], label=f"Train {title}")
            plt.plot(history.history[metric[1]], label=f"Validation {title}")
            plt.legend(); plt.title(f"Training and Validation {title}")
            plt.savefig(os.path.join(args.data_public_out, fname))
            plt.close()


        # predict on the test dataset
        t0 = time.time()
        probs = model.predict(test_df, verbose=0)
        elapsed_ms = (time.time() - t0) * 1000
        print(f"Average inference time per image: {elapsed_ms / len(probs):.2f} ms")

        # Save the results
        csv_path = os.path.join(args.data_out, "test_scores.csv")

        # inj = col 0, noise = col 1

        csv_path = os.path.join(args.data_out, "test_scores.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "Inj_Prob", "Noise_Prob", "Predicted_label"])
        for i, p in enumerate(probs):
            src = file_paths_test[i]
            if isinstance(src, bytes):
                src = src.decode("utf-8")
            inj_p, noise_p = float(p[0]), float(p[1])
            label = "inj" if inj_p >= 0.5 else "noise"
            w.writerow([src, f"{inj_p:.6f}", f"{noise_p:.6f}", label])
 
            if args.save_images:
                # copy of original PNG + metadata 
                dst = os.path.join(args.data_out, f"TTMap_{i}.png")
                shutil.copyfile(src, dst)
                meta = PngInfo()
                combined = read_png_metadata(src)
                combined.update({"Inj_Prob": inj_p, "Noise_Prob": noise_p,
                                 "Predicted_label": label})
                for k, v in combined.items():
                    meta.add_text(k, str(v))
                Image.open(dst).save(dst, "png", pnginfo=meta)
 
    print(f"Scores saved to {csv_path}")

if __name__ == "__main__":
    main()
