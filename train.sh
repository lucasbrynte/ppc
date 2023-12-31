# exit when any command fails
set -e

CONFIGNAME=$1
EXPERIMENT_PREFIX=$2

REPOPATH=/home/lucas/research/ppc
CONTAINER=ppc

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
# OBJECTS=(duck can cat eggbox glue holepuncher)
# Discard driller (not present in validation sequence):
# OBJECTS=(duck can cat eggbox glue holepuncher ape)
# OBJECTS=(duck can cat driller eggbox glue holepuncher ape)


for OBJ in ${OBJECTS[@]}; do
    echo "Removing experiment /hdd/lucas/out/ppc-experiments/$EXPERIMENT_PREFIX/$OBJ"
    rm -rf /hdd/lucas/out/ppc-experiments/$EXPERIMENT_PREFIX/$OBJ
done

xhost + # allow connections to X server

# Temporarily disable exit-on-error, in order to manually handle errors, and force cleanup.
set +e
for OBJ in ${OBJECTS[@]}; do
    echo "CURRENT OBJECT: ""$OBJ"" / (""${OBJECTS[@]}"")"
    ./rundocker.sh \
        $CONTAINER $CMD main.py \
        train \
        --config-name $CONFIGNAME \
        --experiment-name $EXPERIMENT_PREFIX/$OBJ \
        --obj-label $OBJ
    if [ $? -ne 0 ]; then
        echo "Breaking..."
        break
    fi
done
set -e

popd

echo "Cleanup..."
rm -rf $WS
echo "Done."
