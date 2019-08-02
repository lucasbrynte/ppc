# exit when any command fails
set -e

OLD_EXPERIMENT_PREFIX=$1
NEW_EXPERIMENT_PREFIX=$2

REPOPATH=/home/lucas/research/ppc
DATAPATH=/home/lucas/datasets/pose-data/sixd/occluded-linemod-augmented3_format06
CONTAINER=ppc
CONFIGNAME=dummy


TMP_SUFFIX=$(openssl rand -hex 4)
WS=/tmp/ppc-ws-$TMP_SUFFIX
rm -rf $WS
cp -r $REPOPATH $WS


OBJECTS=(duck)
# Discard driller (not present in validation sequence):
# OBJECTS=(duck can cat eggbox glue holepuncher ape)
# OBJECTS=(duck can cat driller eggbox glue holepuncher ape)


for OBJ in ${OBJECTS[@]}; do
    echo "Removing experiment /hdd/lucas/out/3dod-experiments/$NEW_EXPERIMENT_PREFIX/$OBJ"
    rm -rf /hdd/lucas/out/ppc-experiments/$NEW_EXPERIMENT_PREFIX/$OBJ
done

xhost + # allow connections to X server
for OBJ in ${OBJECTS[@]}; do
    # NOTE: ndocker seems to ignore NV_GPU when the X socket is fed into the container anyway.
    # Mimick behavior by setting CUDA_VISIBLE_DEVICES inside container instead.
    ndocker \
        -e PYTHONPATH=/workspace/ppc \
        -e CUDA_VISIBLE_DEVICES=$NV_GPU \
        -e DISPLAY=unix:0.$NV_GPU \
        -v /tmp/.X11-unix:/tmp/.X11-unix --privileged \
        -w /workspace/ppc \
        -v $WS:/workspace/ppc \
        -v /hdd/lucas/out/ppc-experiments:/workspace/ppc/experiments \
        -v $DATAPATH:/datasets/occluded-linemod-augmented \
        $CONTAINER python main.py \
        eval \
        --overwrite-experiment \
        --config-name $CONFIGNAME \
        --experiment-name $NEW_EXPERIMENT_PREFIX/$OBJ \
        --old-experiment-name $OLD_EXPERIMENT_PREFIX/$OBJ \
        --checkpoint-load-fname best_model.pth.tar \
        --obj-label $OBJ
done
rm -rf $WS
