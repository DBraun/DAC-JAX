#!/bin/bash
#SBATCH --job-name=jax-gpu       # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=16G        # RAM usage per cpu-core
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=03:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=END          # choice could be 'fail'
#SBATCH --mail-user=db1224@princeton.edu

module purge
module load anaconda3/2024.02

eval "$(conda shell.bash hook)"
conda activate jax-env

python scripts/benchmark.py --model_type=16khz
python scripts/benchmark.py --model_type=24khz
python scripts/benchmark.py --model_type=44khz
