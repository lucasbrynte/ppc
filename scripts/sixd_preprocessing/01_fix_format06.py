import os
import shutil
import glob

# root_path = '/home/lucas/datasets/pose-data/sixd/lm-lmo-from-bop/v1'
root_path = '/datasets/lm-lmo-from-bop'

png_paths = glob.glob(os.path.join(root_path, '*', '*', '*', '*.png'))
print("Sorting...")
png_paths = sorted(png_paths)
print("Done sorting.")

for j, png_path in enumerate(png_paths):
    if (j+1) % 100 == 0:
        print("{}/{}".format(j+1, len(png_paths)))

    dir_path = os.path.dirname(png_path)
    old_fname = os.path.basename(png_path)
    frame_idx = int(old_fname.split('.')[0])
    new_fname = "{:06d}.png".format(frame_idx)
    if old_fname != new_fname:
        assert not os.path.exists(os.path.join(dir_path, new_fname))
        shutil.move(os.path.join(dir_path, old_fname), os.path.join(dir_path, new_fname))
