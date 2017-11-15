#!/usr/bin/env bash
unset PYTHONPATH
module load plgrid/apps/cuda/8.0
module load plgrid/tools/python/3.6.0
source venv/bin/activate
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/net/people/plgfracz/cudnn/cuda/lib64
