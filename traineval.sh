# # exit when any command fails
# set -e

# # EVALMODE=eval
# EVALMODE=eval_poseopt
EVALMODE=$1
CONFIGNAME=$2
OLD_EXPERIMENT_PREFIX=$3
NEW_EXPERIMENT_PREFIX=$4

REPOPATH=/home/lucas/research/ppc
CONTAINER=ppc
# CHECKPOINT_FNAME=best_model.pth.tar
# CHECKPOINT_FNAME=epoch010.pth.tar
CHECKPOINT_FNAME=latest_model.pth.tar


# Create tmp dir (needs to exist for automatic cleanup to work)
mkdir -p /tmp/ppc-ws
# Remove all but 20 most recent tmp dirs
pushd /tmp/ppc-ws
ls -t1 . | tail -n+20 | xargs rm -rf
popd
TMP_SUFFIX=$(openssl rand -hex 4)
WS=/tmp/ppc-ws/$TMP_SUFFIX
rm -rf $WS
cp -r $REPOPATH $WS
pushd $WS

CMD="python"
# CMD="python -m cProfile -s cumtime"
# CMD="kernprof -l"

OBJECTS=(duck)
# OBJECTS=(can)
# OBJECTS=(duck can cat)
# Discard driller (not present in validation sequence):
# OBJECTS=(duck can cat eggbox glue holepuncher ape)
# OBJECTS=(duck can cat driller eggbox glue holepuncher ape)


for OBJ in ${OBJECTS[@]}; do
    echo "Removing old experiment /hdd/lucas/out/ppc-experiments/$OLD_EXPERIMENT_PREFIX/$OBJ"
    rm -rf /hdd/lucas/out/ppc-experiments/$OLD_EXPERIMENT_PREFIX/$OBJ
    echo "Removing new experiment /hdd/lucas/out/ppc-experiments/$NEW_EXPERIMENT_PREFIX/$OBJ"
    rm -rf /hdd/lucas/out/ppc-experiments/$NEW_EXPERIMENT_PREFIX/$OBJ
done

xhost + # allow connections to X server

# Temporarily disable exit-on-error, in order to manually handle errors, and force cleanup.
set +e
for OBJ in ${OBJECTS[@]}; do
    # TRAIN
    echo "CURRENT OBJECT: ""$OBJ"" / (""${OBJECTS[@]}"")"
    ./rundocker.sh \
        $CONTAINER $CMD main.py \
        train \
        --config-name $CONFIGNAME \
        --experiment-name $OLD_EXPERIMENT_PREFIX/$OBJ \
        --obj-label $OBJ
    if [ $? -ne 0 ]; then
        echo "Breaking..."
        break
    fi
    # EVAL
    echo "CURRENT OBJECT: ""$OBJ"" / (""${OBJECTS[@]}"")"
    ./rundocker.sh \
        $CONTAINER $CMD main.py \
        $EVALMODE \
        --config-name $CONFIGNAME \
        --experiment-name $NEW_EXPERIMENT_PREFIX/$OBJ \
        --old-experiment-name $OLD_EXPERIMENT_PREFIX/$OBJ \
        --checkpoint-load-fname $CHECKPOINT_FNAME \
        --obj-label $OBJ
    if [ $? -ne 0 ]; then
        echo "Breaking..."
        break
    fi
done
popd

echo "Cleanup..."
rm -rf $WS
echo "Done."
