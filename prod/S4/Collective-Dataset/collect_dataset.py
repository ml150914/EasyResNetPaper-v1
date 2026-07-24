import os
import shutil

Eccentricity = '../Eccentricity/injections_16_bins'
HoM =          '../HoM/injections_16_bins'
ExtremeSpin =  '../ExtremeSpin/injections_16_bins'
Precessing =   '../Precessing/injections_16_bins'
Superimposed = '../Superimposed/injections_16_bins'

n_per_folder = 15680

Collected = 'Collected_copied'

os.makedirs(Collected, exist_ok = True)

counter = 1

for folder in [Eccentricity, HoM, ExtremeSpin, Precessing, Superimposed]:
    files = sorted(
        f for f in os.listdir(folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
        )
    files = files[:n_per_folder]
    for filename in files:
        src = os.path.join(folder, filename)
        ext = os.path.splitext(filename)[1]
        dst = os.path.join(Collected, f"TT_map_SNR_{counter}{ext}")

        if os.path.lexists(dst):
            os.remove(dst)

        os.symlink(os.path.abspath(src), dst)
        #shutil.copy2(src, dst)
        print(f'Linked {src} -> {dst}')
        counter += 1 
