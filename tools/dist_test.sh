#!/bin/bash
NUM_PROC=$1
MASTER_PORT=$2
shift
torchrun --nproc_per_node=$NUM_PROC --master_port $MASTER_PORT test.py ${@:2}
