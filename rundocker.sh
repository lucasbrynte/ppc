# exit when any command fails
set -e

# DATAPATH=/home/lucas/datasets/pose-data/sixd/occluded-linemod-augmented5_split_unoccl_train_test
# DATAPATH=/home/lucas/datasets/pose-data/sixd/occluded-linemod-augmented6_decimated_models
DATAPATH=/home/lucas/datasets/pose-data/sixd/occluded-linemod-augmented5b_posecnn_anno
LM_LMO_BOP_PATH=/home/lucas/datasets/pose-data/sixd/lm-lmo-from-bop/v1
LMO_BOP2019_PATH=/home/lucas/datasets/pose-data/sixd/bop2019-lmo
DEEPIMPATH=/home/lucas/datasets/pose-data/deepim-resources
FLOWNETPATH=/home/lucas/datasets/flownet2/pth_models
NYUDPATH=/home/lucas/datasets/nyud
VOCPATH=/home/lucas/datasets/VOC/VOCdevkit/VOC2012


xhost + # allow connections to X server
# NOTE: ndocker seems to ignore NV_GPU when the X socket is fed into the container anyway.
# Mimick behavior by setting CUDA_VISIBLE_DEVICES inside container instead.
docker run \
    -it \
    --ipc host \
    -e HOST_USER_ID=$(id -u) -e HOST_GROUP_ID=$(id -g) \
    -e PYTHONPATH=/workspace \
    -e CUDA_VISIBLE_DEVICES=$NV_GPU \
    -e DISPLAY=unix:0.$NV_GPU \
    -v /tmp/.X11-unix:/tmp/.X11-unix --privileged \
    -w /workspace \
    -v $PWD:/workspace \
    -v /hdd/lucas/out/ppc-experiments:/workspace/experiments \
    -v $DATAPATH:/datasets/occluded-linemod-augmented \
    -v $LM_LMO_BOP_PATH:/datasets/lm-lmo-from-bop \
    -v $LMO_BOP2019_PATH:/datasets/lmo_bop19 \
    -v $DEEPIMPATH:/datasets/deepim-resources \
    -v $FLOWNETPATH:/flownet2_models \
    -v $NYUDPATH:/datasets/nyud \
    -v $VOCPATH:/datasets/voc \
    $@
#    -v /etc/resolv.conf:/etc/resolv.conf \
