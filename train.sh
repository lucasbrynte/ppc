# exit when any command fails
set -e

EXPERIMENT_PREFIX=$1

REPOPATH=/home/lucas/research/ppc
CONTAINER=ppc
CONFIGNAME=dummy

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
# OBJECTS=(duck can cat)
# OBJECTS=(duck can cat eggbox glue holepuncher)
# Discard driller (not present in validation sequence):
# OBJECTS=(duck can cat eggbox glue holepuncher ape)
# OBJECTS=(duck can cat driller eggbox glue holepuncher ape)


for OBJ in ${OBJECTS[@]}; do
    echo "Removing experiment /hdd/lucas/out/ppc-experiments/$EXPERIMENT_PREFIX/$OBJ"
    rm -rf /hdd/lucas/out/ppc-experiments/$EXPERIMENT_PREFIX/$OBJ
done

xhost + # allow connections to X server
for OBJ in ${OBJECTS[@]}; do
    ./rundocker.sh \
        $CONTAINER $CMD main.py \
        train \
        --config-name $CONFIGNAME \
        --experiment-name $EXPERIMENT_PREFIX/$OBJ \
        --obj-label $OBJ
done
popd
rm -rf $WS
