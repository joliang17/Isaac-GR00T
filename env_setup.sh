#!/bin/bash


module load gcc/11.2.0
module load cuda/12.4.1

pip install "git+https://github.com/facebookresearch/pytorch3d.git"
pip install --no-build-isolation flash-attn==2.7.1.post4 

