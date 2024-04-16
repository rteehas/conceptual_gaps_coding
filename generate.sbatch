#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=3:00:00
#SBATCH --mem-per-cpu=16GB
#SBATCH --job-name=TeehanConceptual
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=rst306@nyu.edu
#SBATCH --output=slurm_%j.out

cd /scratch/rst306/conceptual_gaps_coding
singularity exec --nv \
                --overlay /scratch/rst306/hierarchical-transformer/rmt_vae.ext3:ro \
                /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif\
                    /bin/bash -c "source /ext3/env.sh; python3 run_generation.py"
