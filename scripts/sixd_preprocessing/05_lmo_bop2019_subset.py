"""
Split train / test sets according to DeepIM split, and create new directories deepim_train_unoccl & deepim_test_unoccl
"""

import os
import shutil
import json
import ruamel.yaml as yaml

LM_PATH = '/datasets/lm-lmo-from-bop'
LMO_BOP2019_PATH = '/datasets/lmo_bop19'

# DRY_RUN = True
DRY_RUN = False
EXIST_OK = False

all_occl_seq_path = os.path.join(LM_PATH, 'test_occl', 'benchviseblue')
test_bop2019_seq_path = os.path.join(LM_PATH, 'test_bop2019_occl', 'benchviseblue')

def copy_img_subdir_files(src_path, dst_path, old_indices, new_indices):
    os.makedirs(dst_path, exist_ok=True)
    for old_idx, new_idx in zip(old_indices, new_indices):
        old_path = os.path.join(src_path, '{:06d}.png'.format(old_idx))
        new_path = os.path.join(dst_path, '{:06d}.png'.format(new_idx))
        if not DRY_RUN:
            shutil.copyfile(old_path, new_path)

def filter_dict(old_dict, old_indices, new_indices):
    return { new_idx: old_dict[old_idx] for old_idx, new_idx in zip(old_indices, new_indices) }


with open(os.path.join(LMO_BOP2019_PATH, 'test', '000002', 'scene_gt.json'), 'r') as f:
    test_bop2019_gts = json.load(f)
    test_bop2019_gts = { int(frame_idx): anno for frame_idx, anno in test_bop2019_gts.items() }
test_bop2019_frame_indices = sorted(test_bop2019_gts.keys())

if not DRY_RUN:
    os.makedirs(test_bop2019_seq_path, exist_ok=EXIST_OK)

with open(os.path.join(all_occl_seq_path, 'gt.yml'), 'r') as f:
    all_gts = yaml.load(f, Loader=yaml.CLoader)
with open(os.path.join(all_occl_seq_path, 'info.yml'), 'r') as f:
    all_infos = yaml.load(f, Loader=yaml.CLoader)

# Copy global_info.yml
if not DRY_RUN:
    shutil.copyfile(os.path.join(all_occl_seq_path, 'global_info.yml'), os.path.join(test_bop2019_seq_path, 'global_info.yml'))

# Filter & store gt.yml
test_bop2019_gts_orig_lmo_src = filter_dict(all_gts, test_bop2019_frame_indices, test_bop2019_frame_indices)
assert set(test_bop2019_gts.keys()) == set(test_bop2019_gts_orig_lmo_src.keys())
print('Total #annotated objects in BOP2019 challenge: {}'.format(sum([ len(anno) for anno in test_bop2019_gts.values() ])))
print('Total #annotated objects in my data (corresponding to original LM-O?): {}'.format(sum([ len(anno) for anno in test_bop2019_gts_orig_lmo_src.values() ])))
# print(json.dumps(test_bop2019_gts[3][0], indent=True))
# print(json.dumps(test_bop2019_gts_orig_lmo_src[3][0], indent=True))
for frame_idx in test_bop2019_gts_orig_lmo_src:
    obj_ids_orig_lmo_src = { obj_anno['obj_id'] for obj_anno in test_bop2019_gts_orig_lmo_src[frame_idx] }
    obj_ids_bop2019 = { obj_anno['obj_id'] for obj_anno in test_bop2019_gts[frame_idx] }
    assert (obj_ids_bop2019 <= obj_ids_orig_lmo_src), 'Found object annotations in BOP2019 data, which were not present in original LM-O data!'
    # Disregard object annotations according to BOP2019.
    if obj_ids_bop2019 < obj_ids_orig_lmo_src:
        for obj_anno in test_bop2019_gts_orig_lmo_src[frame_idx]:
            if obj_anno['obj_id'] not in obj_ids_bop2019:
                assert obj_anno['obj_id'] == 2 # Actually, this seems to happen only due to missing benchvise annotations in test set (which makes sense). Otherwise object presence seems to be consistent.
                # Disregard object by replacing obj_id by negative dummy obj_id. This way, the instance_seg indices are still valid. Alternatively, they should have been modified.
                obj_anno['obj_id'] -= 100
            else:
                bop2019_obj_anno_matches = [ bop2019_obj_anno for bop2019_obj_anno in test_bop2019_gts[frame_idx] if bop2019_obj_anno['obj_id'] == obj_anno['obj_id'] ]
                assert len(bop2019_obj_anno_matches) == 1
                bop2019_obj_anno = bop2019_obj_anno_matches[0]
                # For peace of mind: ensure all pose annos are the same. Although not absolutely necessary, this seems to be the case.
                assert obj_anno['cam_R_m2c'] == bop2019_obj_anno['cam_R_m2c']
                assert obj_anno['cam_t_m2c'] == bop2019_obj_anno['cam_t_m2c']
# print('Total #annotated objects in BOP2019 challenge: {}'.format(sum([ len(anno) for anno in test_bop2019_gts.values() ])))
print('Total #annotated objects in my data (corresponding to original LM-O?), excluding "dummy" object annos modified just now: {}'.format(sum([ 1 for anno in test_bop2019_gts_orig_lmo_src.values() for obj_anno in anno if obj_anno['obj_id'] > 0 ])))
if not DRY_RUN:
    with open(os.path.join(test_bop2019_seq_path, 'gt.yml'), 'w') as f:
        yaml.dump(test_bop2019_gts_orig_lmo_src, f, Dumper=yaml.CDumper)



# Filter & store info.yml
test_bop2019_infos_orig_lmo_src = filter_dict(all_infos, test_bop2019_frame_indices, test_bop2019_frame_indices)
if not DRY_RUN:
    with open(os.path.join(test_bop2019_seq_path, 'info.yml'), 'w') as f:
        yaml.dump(test_bop2019_infos_orig_lmo_src, f, Dumper=yaml.CDumper)



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
            os.path.join(all_occl_seq_path, img_subdir),
            os.path.join(test_bop2019_seq_path, img_subdir),
            test_bop2019_frame_indices,
            test_bop2019_frame_indices,
        )
print('Done!')
