# exit when any command fails
set -e

EXPERIMENT_PREFIX=$1

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
    echo "Removing experiment /hdd/lucas/out/3dod-experiments/$EXPERIMENT_PREFIX/$OBJ"
    rm -rf /hdd/lucas/out/ppc-experiments/$EXPERIMENT_PREFIX/$OBJ
done

xhost + # allow connections to X server
for OBJ in ${OBJECTS[@]}; do
    ndocker \
        -e PYTHONPATH=/workspace/ppc \
        -e DISPLAY=unix:0.0 \
        -v /tmp/.X11-unix:/tmp/.X11-unix --privileged \
        -w /workspace/ppc \
        -v $WS:/workspace/ppc \
        -v /hdd/lucas/out/ppc-experiments:/workspace/ppc/experiments \
        -v $DATAPATH:/datasets/occluded-linemod-augmented \
        $CONTAINER python train.py \
        --overwrite-experiment \
        --config-name $CONFIGNAME \
        --experiment-name $EXPERIMENT_PREFIX/$OBJ \
        --obj-label $OBJ
done
rm -rf $WS