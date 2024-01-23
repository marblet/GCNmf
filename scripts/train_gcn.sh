#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=05:00:00
#SBATCH --output=GCN_training_mf%A.out

source activate thesis
# source scripts/preamble.sh

srun python3 run_node_cls.py --name gcn_graph_run_mf_sep_obs_005 --run_mf --lr 0.005 --epoch 200 --batch_size 256 --nhid 64 --dropout 0.1