#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=08:00:00
#SBATCH --output=GCN_training%A.out

source activate thesis
# source scripts/preamble.sh

srun python3 run_node_cls.py --name gcn_obs_sep_nhid32_4lay --epoch 200 --batch_size 256 --nhid 64 --n_layers 4 --dropout 0.1