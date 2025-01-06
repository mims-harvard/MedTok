#! /bin/bash
#SBATCH --job-name=medical_code
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_mzitnik_lab
#SBATCH --mem=250G
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=16
#SBATCH --time=0-72:00

module load ncf/1.0.0-fasrc01
module load miniconda3/py39_4.11.0-linux_x64-ncf
source activate plm


WORLD_SIZE=4 CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port 1234 train.py