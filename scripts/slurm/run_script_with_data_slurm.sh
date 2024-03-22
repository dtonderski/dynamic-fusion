#!/bin/bash
#SBATCH --partition=shared
#SBATCH --exclude=destc0strapp02
#SBATCH --gres=gpu:1,gpu_mem:16000
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=st_unfold_t_interp
#SBATCH --output=slurm-logs/output.%j.%N.log
#SBATCH --error=slurm-logs/error.%j.%N.log

# Just display some information about NVCC (can be removed)
nvcc --version
echo "\$nvcc --version"
nvidia-smi
echo "\$nvidia-smi"
echo $CUDA_VISIBLE_DEVICES

# echo $SLURM_SCRATCH
export DATA_FOLDER=$SLURM_SCRATCH"/data/interim/coco/"
echo "Data folder is "$DATA_FOLDER

echo "Copying dataset"
mkdir -p $DATA_FOLDER                                                    # Create the folder where the data will be stored

rsync -ah --info=progress2 /cig/cig04b/students/chtonded/data/interim/coco/ $DATA_FOLDER
echo "Dataset copied!"

#cp -r ~/data/interim/coco/2subbins_new/ $DATA_FOLDER                  # Copy the source data tarball (recommended) to the data folder on /disk1 partition

singularity exec --bind $DATA_FOLDER:/mnt,/cig/cig04b/students/chtonded/runs:/runs --nv singularity/python39.sif sh scripts/slurm/run_script_singularity.sh "$@"
