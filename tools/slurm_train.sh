#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")
job='cityscapes_deeplabv3plus_resnet50'

PARTITION=$1
CONFIG=$2
PORT=$3
SAVE_PATH=$4
GPUS=${GPUS:-4}
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
CPUS_PER_TASK=${CPUS_PER_TASK:-5}
SRUN_ARGS=${SRUN_ARGS:-""}
PY_ARGS=${@:4}

mkdir -p ${SAVE_PATH}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \

srun -p ${PARTITION} \
    --job-name=$job \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    --quotatype=auto \
    --preempt \
    --comment=wbsR-SC221442.001 \
    ${SRUN_ARGS} \
    python -u train.py \
    --config=${CONFIG} --save-path ${SAVE_PATH} --port ${PORT} 2>&1 | tee ${SAVE_PATH}/$now.txt