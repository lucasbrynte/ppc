# exit when any command fails
set -e

OLD_EXPERIMENT_PREFIX=$1
NEW_EXPERIMENT_PREFIX=$2

REPOPATH=/home/lucas/research/ppc
CONTAINER=ppc
CONFIGNAME=dummy
# CHECKPOINT_FNAME=best_model.pth.tar
# CHECKPOINT_FNAME=epoch010.pth.tar
CHECKPOINT_FNAME=latest_model.pth.tar
# EVALMODE=eval
EVALMODE=eval_poseopt


TMP_SUFFIX=$(openssl rand -hex 4)
WS=/tmp/ppc-ws-$TMP_SUFFIX
rm -rf $WS
cp -r $REPOPATH $WS
pushd $WS

# PROFILE_ARGS="-m cProfile -s cumtime"

OBJECTS=(duck)
# OBJECTS=(duck can cat)
# Discard driller (not present in validation sequence):
# OBJECTS=(duck can cat eggbox glue holepuncher ape)
# OBJECTS=(duck can cat driller eggbox glue holepuncher ape)


for OBJ in ${OBJECTS[@]}; do
    echo "Removing experiment /hdd/lucas/out/3dod-experiments/$NEW_EXPERIMENT_PREFIX/$OBJ"
    rm -rf /hdd/lucas/out/ppc-experiments/$NEW_EXPERIMENT_PREFIX/$OBJ
done

xhost + # allow connections to X server
for OBJ in ${OBJECTS[@]}; do
    ./rundocker.sh \
        $CONTAINER python $PROFILE_ARGS main.py \
        $EVALMODE \
        --overwrite-experiment \
        --config-name $CONFIGNAME \
        --experiment-name $NEW_EXPERIMENT_PREFIX/$OBJ \
        --old-experiment-name $OLD_EXPERIMENT_PREFIX/$OBJ \
        --checkpoint-load-fname $CHECKPOINT_FNAME \
        --obj-label $OBJ
done
popd
rm -rf $WS
