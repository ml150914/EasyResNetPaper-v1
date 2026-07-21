import os
import shutil

Eccentricity = '../Eccentricity/injections_16_bins'
HoM =          '../HoM/injections_16_bins'
ExtremeSpin =  '../ExtremeSpin/injections_16_bins'

Collected = 'Collected'

os.makedirs(Collected, exist_ok = True)

counter = 1

for folder in [Eccentricity, HoM, ExtremeSpin]:
    files = sorted(
        f for f in os.listdir(folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
        )
    for filename in files:
        src = os.path.join(folder, filename)
        ext = os.path.splitext(filename)[1]
        dst = os.path.join(Collected, f"TT_map_SNR_{counter}{ext}")

        if os.path.lexists(dst):
            os.remove(dst)

        os.symlink(src, dst)
        print(f'Linked {src} -> {dst}')
        counter += 1 
