#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")

CONFIG=$1
PORT=$2
SAVE_PATH=$3
GPUS=$4

mkdir -p ${SAVE_PATH}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \

python -m torch.distributed.launch \
    --nproc_per_node=${GPUS} \
    --master_addr=localhost \
    --master_port=${PORT} \
    train.py \
    --config=${CONFIG} --save-path ${SAVE_PATH} --port ${PORT} 2>&1 | tee ${SAVE_PATH}/$now.txt