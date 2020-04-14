"""
Render ground truth instance segmentations, correspondence maps and normal maps from pose annotations.
"""

import sys
import os
# Add parent directory to python path, to find libraries:
# sys.path.append('../..') # Relative to CWD
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))) # Relative to module, but cannot be used in notebooks

from lib.utils import listdir_nohidden
import yaml
import numpy as np
import shutil


DRY_RUN = False
LINEMOD_FLAG = False

if LINEMOD_FLAG:
    SIXD_PATH = '/home/lucas/datasets/pose-data/sixd/occluded-linemod-augmented2bb_instance_idx_fix'
    SUBSETS = [subset for subset in listdir_nohidden(SIXD_PATH) if subset.startswith('train') or subset.startswith('test')]
else:
    SIXD_PATH = '/home/lucas/datasets/pose-data/sixd/ycb-video'
    SUBSETS = ['data']


def write_pose_tud(pose_path, R, t):
    assert R.shape == (3,3)
    assert t.shape == (3,1)
    D = np.diag([1,-1,-1]);
    R = np.dot(D, R)
    t = np.dot(D, t)
    os.makedirs(os.path.dirname(pose_path), exist_ok=True)
    with open(pose_path, 'w+') as f:
        lines = []
        lines.append("image size ")
        lines.append("640 480")
        lines.append("999") # Dummy object ID
        lines.append("rotation: ")
        for row in R:
            lines.append(' '.join(map(str, row)))
        lines.append("center: ")
        lines.append(' '.join(map(str, t.T[0])))
        lines.append("extent: ") # Dummy extent values
        lines.append(' '.join(map(str, [0.0, 0.0, 0.0])))
        f.writelines([line+'\n' for line in lines])

def read_yaml(path):
    with open(path, 'r') as f:
        return yaml.load(f, Loader=yaml.CLoader)

models_info = read_yaml(os.path.join(SIXD_PATH, 'models', 'models_info.yml'))

for subset in SUBSETS:
    for seq in sorted(listdir_nohidden(os.path.join(SIXD_PATH, subset))):
        render_instance_seg = False if LINEMOD_FLAG and subset == 'train_aug' else True
        render_corr = True
        render_normals = True if LINEMOD_FLAG else False
        render_depth = True if LINEMOD_FLAG else False
        render_rgb = True if LINEMOD_FLAG else False

        rgb_dir = os.path.join(SIXD_PATH, subset, seq, 'rgb')
        instance_seg_dir = os.path.join(SIXD_PATH, subset, seq, 'instance_seg')
        corr_dir = os.path.join(SIXD_PATH, subset, seq, 'obj')
        normals_dir = os.path.join(SIXD_PATH, subset, seq, 'normals')
        depth_rendered_dir = os.path.join(SIXD_PATH, subset, seq, 'depth_rendered')
        rgb_rendered_dir = os.path.join(SIXD_PATH, subset, seq, 'rgb_rendered')

        if not DRY_RUN:
            if render_instance_seg:
                if os.path.exists(instance_seg_dir):
                    shutil.rmtree(instance_seg_dir)
                os.makedirs(instance_seg_dir)
            if render_corr:
                if os.path.exists(corr_dir):
                    shutil.rmtree(corr_dir)
                os.makedirs(corr_dir)
            if render_normals:
                if os.path.exists(normals_dir):
                    shutil.rmtree(normals_dir)
                os.makedirs(normals_dir)
            if render_depth:
                if os.path.exists(depth_rendered_dir):
                    shutil.rmtree(depth_rendered_dir)
                os.makedirs(depth_rendered_dir)
            if render_rgb:
                if os.path.exists(rgb_rendered_dir):
                    shutil.rmtree(rgb_rendered_dir)
                os.makedirs(rgb_rendered_dir)

        gts = read_yaml(os.path.join(SIXD_PATH, subset, seq, 'gt.yml'))

        fnames = list(sorted(listdir_nohidden(rgb_dir)))
        for j, fname in enumerate(fnames):
            img_idx = int(fname.split('.')[0])

            if (j+1) % 100 == 0:
                print("subset {}, seq {}, frame {}/{}".format(subset, seq, j+1, len(fnames)))

            instance_seg_path = os.path.join(instance_seg_dir, fname) if render_instance_seg else '/dev/null'
            corr_path = os.path.join(corr_dir, fname) if render_corr else '/dev/null'
            normals_path = os.path.join(normals_dir, fname) if render_normals else '/dev/null'
            depth_rendered_path = os.path.join(depth_rendered_dir, fname) if render_depth else '/dev/null'
            rgb_rendered_path = os.path.join(rgb_rendered_dir, fname) if render_rgb else '/dev/null'

            tmp_pose_dir = '/tmp/tmp_poses_for_rendering'
            if not DRY_RUN:
                if os.path.exists(tmp_pose_dir):
                    shutil.rmtree(tmp_pose_dir)
                os.makedirs(tmp_pose_dir)

            instance_idx_list = []
            pose_path_list = []
            mesh_path_list = []
            instance_idx = 0
            for gt in gts[img_idx]:
                instance_idx += 1
                mesh_path = os.path.join(SIXD_PATH, 'models', 'obj_{:02d}.ply'.format(gt['obj_id']))

                R = np.array(gt['cam_R_m2c']).reshape((3, 3))
                t = np.array(gt['cam_t_m2c']).reshape((3,1))# * 1e-3
                pose_path = os.path.join(tmp_pose_dir, '{:03d}'.format(instance_idx))

                if not DRY_RUN or True:
                    write_pose_tud(pose_path, R, t)

                instance_idx_list.append(instance_idx)
                pose_path_list.append(pose_path)
                mesh_path_list.append(mesh_path)

            cmd_list = []
            cmd_list.append("/home/lucas/poserenderer/render-gt")
            cmd_list.append("--seg-path")
            cmd_list.append(instance_seg_path)
            cmd_list.append("--coord-path")
            cmd_list.append(corr_path)
            cmd_list.append("--normals-path")
            cmd_list.append(normals_path)
            cmd_list.append("--depth-path")
            cmd_list.append(depth_rendered_path)
            cmd_list.append("--rgb-path")
            cmd_list.append(rgb_rendered_path)
            cmd_list.append("--mesh-paths")
            cmd_list.append(",".join(map(str, mesh_path_list)))
            cmd_list.append("--object-ids")
            cmd_list.append(",".join(map(str, instance_idx_list)))
            cmd_list.append("--pose-paths")
            cmd_list.append(",".join(pose_path_list))

            cmd = ' '.join(cmd_list)


            if not DRY_RUN:
                exit_code = os.system(cmd)
                if exit_code != 0:
                    print("Fail!")
                    sys.exit(exit_code)
