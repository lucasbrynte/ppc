import os
import yaml
import glob

# NOTE! Old code, untested lately. Did this step manually last time...

root_path = '/home/lucas/datasets/pose-data/sixd/ycb-video2'

with open(os.path.join(root_path, 'models', 'models_info.yml'), 'r') as f:
    models_info = yaml.load(f, Loader=yaml.CLoader)

seq_paths = sorted(map(lambda gt_path: os.path.dirname(gt_path), glob.glob(os.path.join(root_path, '*', '*', 'gt.yml'))))

for seq_path in seq_paths:
    print(seq_path)

    with open(os.path.join(seq_path, 'gt.yml'), 'r') as f:
        gt_yaml = yaml.load(f, Loader=yaml.CLoader)

    obj_annotated_and_present = set()
    for frame_idx, frame_anno in gt_yaml.items():
        for instance_anno in frame_anno:
            obj_annotated_and_present.add(instance_anno['obj_id'])

    obj_annotated_and_present = sorted([models_info[obj_id]['readable_label'] for obj_id in obj_annotated_and_present])

    global_info_yaml = {
        'obj_annotated_and_present': obj_annotated_and_present,
    }

    with open(os.path.join(seq_path, 'global_info.yml'), 'w') as f:
        yaml.dump(global_info_yaml, f, Dumper=yaml.CDumper)
