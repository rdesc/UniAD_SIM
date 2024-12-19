#!/usr/bin/zsh

source ~/.zshrc
conda activate uniad

UNIAD_PATH=/nas/users/hyzhou/PAMI2024/release/UniAD_Sim
cd ${UNIAD_PATH}
echo ${PWD}
CUDA_VISIBLE_DEVICES=${1} python tools/closeloop/e2e.py \
    projects/configs/stage2_e2e/base_e2e.py ckpts/uniad_base_e2e.pth \
    --launcher none --eval bbox --tmpdir tmp --output $2
cd -

# conda deactivate uniad