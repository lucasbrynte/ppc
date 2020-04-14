"""
Split train / test sets according to DeepIM split, and create new directories deepim_train_unoccl & deepim_test_unoccl
"""

import os
import shutil
import ruamel.yaml as yaml

# LM_PATH = '/home/lucas/datasets/pose-data/sixd/occluded-linemod-augmented5_split_unoccl_train_test'
# DEEPIM_IMAGE_SET_PATH = '/home/lucas/datasets/pose-data/deepim-resources/data/LINEMOD_6D/LM6d_converted/LM6d_refine/image_set'
# LM_PATH = '/linemod'
# DEEPIM_IMAGE_SET_PATH = '/deepim_image_set'
LM_PATH = '/datasets/lm-lmo-from-bop'
DEEPIM_IMAGE_SET_PATH = '/datasets/deepim-resources/data/LINEMOD_6D/LM6d_converted/LM6d_refine/image_set'

DRY_RUN = True
# DRY_RUN = False
EXIST_OK = False

KEEP_FRAME_INDICES = True
# KEEP_FRAME_INDICES = False

# objects = ['ape']
# objects = ['ape', 'can', 'cat', 'driller', 'duck', 'eggbox', 'glue', 'holepuncher']
objects = [
    'ape',
    'benchviseblue',
    # 'bowl',
    'cam',
    'can',
    'cat',
    # 'cup',
    'driller',
    'duck',
    'eggbox',
    'glue',
    'holepuncher',
    'iron',
    'lamp',
    'phone',
]
deepim_obj_label_lookup = {
    'ape': 'ape',
    'benchviseblue': 'benchvise',
    'cam': 'camera',
    'can': 'can',
    'cat': 'cat',
    'driller': 'driller',
    'duck': 'duck',
    'eggbox': 'eggbox',
    'glue': 'glue',
    'holepuncher': 'holepuncher',
    'iron': 'iron',
    'lamp': 'lamp',
    'phone': 'phone',
}

all_path = os.path.join(LM_PATH, 'all_unoccl')
train_path = os.path.join(LM_PATH, 'deepim_train_unoccl')
test_path = os.path.join(LM_PATH, 'deepim_test_unoccl')

# # Rename train_unoccl -> all_unoccl
# old_train_path = os.path.join(LM_PATH, 'deepim_train_unoccl')
# if not os.path.exists(all_path):
#     print("Renaming train_unoccl -> all_unoccl...")
#     if not DRY_RUN:
#         shutil.move(old_train_path, all_path)

def parse_frame_idx(line):
    rel_path = line.split(' ')[0] # Extract path for ref img
    fname = os.path.basename(rel_path)
    frame_idx = int(fname) - 1 # Index from 0 instead of 1
    return frame_idx

def copy_img_subdir_files(src_path, dst_path, old_indices, new_indices):
    os.makedirs(dst_path, exist_ok=EXIST_OK)
    for old_idx, new_idx in zip(old_indices, new_indices):
        old_path = os.path.join(src_path, '{:06d}.png'.format(old_idx))
        new_path = os.path.join(dst_path, '{:06d}.png'.format(new_idx))
        if not DRY_RUN:
            shutil.copyfile(old_path, new_path)

def filter_dict(old_dict, old_indices, new_indices):
    return { new_idx: old_dict[old_idx] for old_idx, new_idx in zip(old_indices, new_indices) }


for obj in objects:
    print("{}...".format(obj))
    with open(os.path.join(DEEPIM_IMAGE_SET_PATH, 'train_{}.txt'.format(deepim_obj_label_lookup[obj])), 'r') as f:
        all_train_frames = { parse_frame_idx(line) for line in f }
    with open(os.path.join(all_path, obj, 'gt.yml'), 'r') as f:
        all_gts = yaml.load(f, Loader=yaml.CLoader)
    with open(os.path.join(all_path, obj, 'info.yml'), 'r') as f:
        all_infos = yaml.load(f, Loader=yaml.CLoader)
    all_frames_sorted = sorted(all_gts.keys())
    # NOTE: since only frame indices already in all_gts.keys() are considered, the very last frame will remain lost (it got lost in some previous pre-preprocessing step).
    # UPDATE: Should not be a problem anymore.
    train_idx_old = [ frame_idx for frame_idx in all_frames_sorted if frame_idx in all_train_frames ]
    test_idx_old = [ frame_idx for frame_idx in all_frames_sorted if frame_idx not in all_train_frames ]
    if KEEP_FRAME_INDICES:
        train_idx_new = train_idx_old
        test_idx_new = test_idx_old
    else:
        train_idx_new = list(range(len(train_idx_old)))
        test_idx_new = list(range(len(test_idx_old)))

    # Filter all data in all_unoccl based on train/test split, and move to deepim_train_unoccl or test_occl
    if not DRY_RUN:
        os.makedirs(os.path.join(train_path, obj), exist_ok=EXIST_OK)
        os.makedirs(os.path.join(test_path, obj), exist_ok=EXIST_OK)

    # Copy global_info.yml
    if not DRY_RUN:
        shutil.copyfile(os.path.join(all_path, obj, 'global_info.yml'), os.path.join(train_path, obj, 'global_info.yml'))
        shutil.copyfile(os.path.join(all_path, obj, 'global_info.yml'), os.path.join(test_path, obj, 'global_info.yml'))

    # Filter & copy gt.yml
    train_gts = filter_dict(all_gts, train_idx_old, train_idx_new)
    test_gts = filter_dict(all_gts, test_idx_old, test_idx_new)
    if not DRY_RUN:
        with open(os.path.join(train_path, obj, 'gt.yml'), 'w') as f:
            yaml.dump(train_gts, f, Dumper=yaml.CDumper)
        with open(os.path.join(test_path, obj, 'gt.yml'), 'w') as f:
            yaml.dump(test_gts, f, Dumper=yaml.CDumper)

    # Filter & copy info.yml
    train_infos = filter_dict(all_infos, train_idx_old, train_idx_new)
    test_infos = filter_dict(all_infos, test_idx_old, test_idx_new)
    if not DRY_RUN:
        with open(os.path.join(train_path, obj, 'info.yml'), 'w') as f:
            yaml.dump(train_infos, f, Dumper=yaml.CDumper)
        with open(os.path.join(test_path, obj, 'info.yml'), 'w') as f:
            yaml.dump(test_infos, f, Dumper=yaml.CDumper)

    # Filter & copy img files
    for img_subdir in [
        'depth',
        'depth_rendered',
        'instance_seg',
        'normals',
        'obj',
        'rgb',
        'rgb_rendered',
        'seg',
    ]:
        if not DRY_RUN:
            copy_img_subdir_files(
                os.path.join(all_path, obj, img_subdir),
                os.path.join(train_path, obj, img_subdir),
                train_idx_old,
                train_idx_new,
            )
            copy_img_subdir_files(
                os.path.join(all_path, obj, img_subdir),
                os.path.join(test_path, obj, img_subdir),
                test_idx_old,
                test_idx_new,
            )
