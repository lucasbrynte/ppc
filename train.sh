# exit when any command fails
set -e

EXPERIMENT_PREFIX=$1

REPOPATH=/home/lucas/research/ppc
# DATAPATH=/home/lucas/datasets/pose-data/sixd/occluded-linemod-augmented3_format06
CONTAINER=ppc
CONFIGNAME=dummy


TMP_SUFFIX=$(openssl rand -hex 4)
WS=/tmp/ppc-ws-$TMP_SUFFIX
rm -rf $WS
cp -r $REPOPATH $WS


ndocker \
    -e PYTHONPATH=/workspace/ppc \
    -w /workspace/ppc \
    -v $WS:/workspace/ppc \
    -v /hdd/lucas/out/ppc-experiments:/workspace/ppc/experiments \
    $CONTAINER python train.py \
    --overwrite-experiment \
    --config-name $CONFIGNAME \
    --experiment-name $EXPERIMENT_PREFIX

rm -rf $WS
