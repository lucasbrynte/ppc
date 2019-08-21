# exit when any command fails
set -e

EXPERIMENT_PREFIX=$1

REPOPATH=/home/lucas/research/ppc
DATAPATH=/home/lucas/datasets/pose-data/sixd/occluded-linemod-augmented3_format06
NYUDPATH=/home/lucas/datasets/nyud
CONTAINER=ppc
CONFIGNAME=dummy


TMP_SUFFIX=$(openssl rand -hex 4)
WS=/tmp/ppc-ws-$TMP_SUFFIX
rm -rf $WS
cp -r $REPOPATH $WS

# PROFILE_ARGS="-m cProfile -s cumtime"

OBJECTS=(duck)
# OBJECTS=(duck can cat)
# OBJECTS=(duck can cat eggbox glue holepuncher)
# Discard driller (not present in validation sequence):
# OBJECTS=(duck can cat eggbox glue holepuncher ape)
# OBJECTS=(duck can cat driller eggbox glue holepuncher ape)


for OBJ in ${OBJECTS[@]}; do
    echo "Removing experiment /hdd/lucas/out/3dod-experiments/$EXPERIMENT_PREFIX/$OBJ"
    rm -rf /hdd/lucas/out/ppc-experiments/$EXPERIMENT_PREFIX/$OBJ
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
        -v $NYUDPATH:/datasets/nyud \
        $CONTAINER python $PROFILE_ARGS main.py \
        train \
        --overwrite-experiment \
        --config-name $CONFIGNAME \
        --experiment-name $EXPERIMENT_PREFIX/$OBJ \
        --obj-label $OBJ
done
rm -rf $WS
