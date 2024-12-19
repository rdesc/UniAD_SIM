# UniAD End-to-End Inference 

This repo is the implementation of UniAD test client for [HUGSIM benchmark](https://xdimlab.github.io/HUGSIM/)

The implementation is based on:
> [**Planning-oriented Autonomous Driving**](https://arxiv.org/abs/2212.10156)

>[arXiv Paper](https://arxiv.org/abs/2212.10156), CVPR 2023 Best Paper

# Installation

Please refer to [UniAD](https://github.com/OpenDriveLab/UniAD) and [VAD](https://github.com/hustvl/VAD) installation instructions. In practice, UniAD and VAD can share the same conda environment.

Please change ${UniAD_PATH} in tools/e2e.sh as the path on your machine.

# Launch Client

### Manually Launch
``` bash
zsh ./tools/e2e.sh ${CUDA_ID} ${output_dir}
```

### Auto Lauch
Client can be auto lauched by the HUGSIM closed-loop script.