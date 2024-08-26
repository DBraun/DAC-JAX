#!/bin/bash
#SBATCH --job-name=DAC-JAX       # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=2        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=16G        # RAM usage per cpu-core
#SBATCH --gres=gpu:2             # number of gpus per node
#SBATCH --time=60:00:00          # total run time limit (HH:MM:SS)
#SBATCH --signal=B:USR1@120      # 120 sec grace period for cleanup after timeout
#SBATCH --signal=B:SIGTERM@120   # 120 sec grace period for cleanup after scancel is sent
#SBATCH --mail-type=END          # choice could be 'fail'
#SBATCH --mail-user=db1224@princeton.edu

function cleanup() {
    echo 'Running cleanup script'
    kill $TRAIN_PID
    kill $TB_PID
    cp -r "/scratch/$USER/runs" "/n/fs/audiovis/$USER/DAC-JAX/runs"
    rm -rf "/scratch/$USER"
    exit 0
}

## Trap the SIGTERM signal (sent by scancel) and call the cleanup function
trap cleanup EXIT SIGINT SIGTERM

module purge
module load anaconda3/2024.02

eval "$(conda shell.bash hook)"

conda activate ../Terrapin/.env/jax-env
export PYTHONPATH=$PWD

## prepare data
##echo "$(date '+%H:%M:%S'): Copying data to /scratch"
##mkdir -p "/scratch/$USER/datasets"
##rsync -a --info=progress2 --no-i-r "/n/fs/audiovis/$USER/datasets/nsynth" "/scratch/$USER/datasets"
#
##cd "/scratch/$USER/datasets/nsynth" || exit
##echo "$(date '+%H:%M:%S'): Unzipping test"
##tar -xzf nsynth-test.jsonwav.tar.gz
##echo "$(date '+%H:%M:%S'): Unzipping valid"
##tar -xzf nsynth-valid.jsonwav.tar.gz
##echo "$(date '+%H:%M:%S'): Unzipping train"
##tar -xzf nsynth-train.jsonwav.tar.gz
##echo "$(date '+%H:%M:%S'): Copied data to /scratch"

## prepare data
echo "$(date '+%H:%M:%S'): Copying data to /scratch"
mkdir -p "/scratch/$USER/datasets"
rsync -a --info=progress2 --no-i-r "/n/fs/audiovis/$USER/datasets/musdb18hq" "/scratch/$USER/datasets"

cd "/scratch/$USER/datasets/musdb18hq" || exit
echo "$(date '+%H:%M:%S'): Unzipping musdb18hq"
unzip musdb18hq.zip
echo "$(date '+%H:%M:%S'): Copied data to /scratch"

## Launch TensorBoard and get the process ID of TensorBoard
tensorboard --logdir="/scratch/$USER/runs" --port=10013 --samples_per_plugin audio=20 --bind_all & TB_PID=$!

cd "/n/fs/audiovis/$USER/DAC-JAX" || exit
python scripts/train.py \
  --args.load conf/neuronic.yml \
  --train.name="slurm_$SLURM_JOB_ID" \
  --train.ckpt_dir="/scratch/$USER/runs" \
  & TRAIN_PID=$!

wait $TRAIN_PID
