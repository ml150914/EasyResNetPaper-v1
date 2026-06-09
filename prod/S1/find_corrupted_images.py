#!/usr/bin/env python
"""
Trova le immagini corrotte/troncate che bloccano la pipeline tf.data.

Usa lo STESSO decoder di TensorFlow, quindi cattura esattamente i file che
farebbero fallire model.fit (es. 'Invalid PNG data').

Uso:
    python find_bad_images.py dataset                 # solo report
    python find_bad_images.py dataset --delete        # elimina i file rotti
    python find_bad_images.py dataset --min-bytes 2048
"""

import os, argparse, warnings
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore")
import tensorflow as tf

EXTS = (".png", ".jpg", ".jpeg")


def check(path):
    """Ritorna None se l'immagine si decodifica, altrimenti il messaggio d'errore."""
    try:
        data = tf.io.read_file(path)
        tf.io.decode_image(data, channels=1, expand_animations=False)
        return None
    except Exception as e:
        # il messaggio TF e' tipo "{{function_node ...}} Invalid PNG data... [Op:...]"
        msg = str(e).replace("\n", " ")
        if "}}" in msg:
            msg = msg.split("}}", 1)[1]
        msg = msg.split("[Op:", 1)[0]
        return msg.strip()[:90] or "decode error"


def main():
    ap = argparse.ArgumentParser(description="Find corrupt images that break tf.data")
    ap.add_argument("root", help="cartella del dataset (scansione ricorsiva)")
    ap.add_argument("--delete", action="store_true", help="elimina i file problematici")
    ap.add_argument("--min-bytes", type=int, default=1024,
                    help="segnala anche i file piu' piccoli di N byte (default 1024)")
    args = ap.parse_args()

    bad, total = [], 0
    for dirpath, _, files in os.walk(args.root):
        for f in files:
            if not f.lower().endswith(EXTS):
                continue
            p = os.path.join(dirpath, f)
            total += 1
            size = os.path.getsize(p)
            err = check(p)
            if err or size < args.min_bytes:
                bad.append((p, size, err or f"troppo piccolo ({size} B)"))

    print(f"Scansionati {total} file -> {len(bad)} problematici")
    for p, size, err in bad:
        print(f"  [{size:>9d} B] {p} :: {err}")

    if bad and args.delete:
        for p, _, _ in bad:
            os.remove(p)
        print(f"\nEliminati {len(bad)} file. Se hai cancellato dal dataset gia' "
              f"splittato, valuta di rigenerare lo split per ribilanciare le classi.")
    elif bad:
        print("\nRiesegui con --delete per rimuoverli.")
    else:
        print("Nessun file corrotto. Il problema potrebbe essere altrove.")


if __name__ == "__main__":
    main()
