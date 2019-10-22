# exit when any command fails
set -e

EXPERIMENT_PREFIX=$1

REPOPATH=/home/lucas/research/ppc
CONTAINER=ppc
CONFIGNAME=dummy


TMP_SUFFIX=$(openssl rand -hex 4)
WS=/tmp/ppc-ws-$TMP_SUFFIX
rm -rf $WS
cp -r $REPOPATH $WS
pushd $WS

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
    ./rundocker.sh \
        $CONTAINER python $PROFILE_ARGS main.py \
        train \
        --overwrite-experiment \
        --config-name $CONFIGNAME \
        --experiment-name $EXPERIMENT_PREFIX/$OBJ \
        --obj-label $OBJ
done
popd
rm -rf $WS
