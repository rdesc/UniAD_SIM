#!/bin/bash

# NOTE: the output directory has to be the same as the one used by the sim server launched by closed_loop.py !!!!
# The client and server communicate using files `obs_pipe` and `plan_pipe` in the output directory.

source ~/.bashrc

# option 1: conda
# conda activate uniad

# UNIAD_PATH=$HUGSIM_WORKSPACE/UniAD_SIM
# cd ${UNIAD_PATH}
# echo ${PWD}
# python tools/closeloop/e2e.py \
#     projects/configs/stage2_e2e/base_e2e.py ckpts/uniad_base_e2e.pth \
#     --launcher none --eval bbox --tmpdir tmp --output output/run-1

# option 2: singularity
singularity exec --nv \
  --bind /home/mila/d/deschaer/hugsim_workspace/UniAD_SIM/:/UniAD_SIM \
  --bind /home/mila/d/deschaer/scratch/nuscenes/:/home/mila/d/deschaer/scratch/nuscenes/ \
  --bind /home/mila/d/deschaer/scratch/hugsim_data/sample_data/sample_data/model/scene-0383/benchmark/uniad/scene-0383_hard_00:/scene-0383_hard_00 \
  --pwd /UniAD_SIM \
  /home/mila/d/deschaer/scratch/singularity_images/uniad.sif \
  python -u tools/closeloop/e2e.py \
    projects/configs/stage2_e2e/base_e2e.py ckpts/uniad_base_e2e.pth \
    --launcher none --eval bbox --tmpdir tmp --output /scene-0383_hard_00

