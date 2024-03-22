#!/bin/bash
#SBATCH --partition=shared
#SBATCH --exclude=destc0strapp02
#SBATCH --gres=gpu:1,gpu_mem:16000
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=generate_data
#SBATCH --output=slurm-logs/output.%j.%N.log
#SBATCH --error=slurm-logs/error.%j.%N.log

# Just display some information about NVCC (can be removed)
echo "\$nvcc --version"
nvcc --version

echo "\$nvidia-smi"
nvidia-smi

echo $CUDA_VISIBLE_DEVICES

singularity exec --bind /cig/cig04b/students/chtonded/data:/mnt --nv singularity/python39.sif sh scripts/slurm/generate_data_singularity.sh
