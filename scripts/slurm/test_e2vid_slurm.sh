#!/bin/bash
#SBATCH --partition=shared
#SBATCH --gres=gpu:1,gpu_mem:30000
#SBATCH --mem=30G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=test
#SBATCH --output=slurm-logs/output.%j.%N.log
#SBATCH --error=slurm-logs/error.%j.%N.log

# Just display some information about NVCC (can be removed)
echo "\$nvcc --version"
nvcc --version

echo "\$nvidia-smi"
nvidia-smi

echo $CUDA_VISIBLE_DEVICES

singularity exec --nv singularity/python39.sif sh scripts/slurm/test_e2vid_singularity.sh
