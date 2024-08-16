#!/bin/bash
#SBATCH --job-name=MusicGen      # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=4        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # RAM usage per cpu-core
#SBATCH --gres=gpu:2             # number of gpus per node
#SBATCH --time=96:00:00          # total run time limit (HH:MM:SS)
#SBATCH --signal=B:USR1@120      # 120 sec grace period for cleanup after timeout
#SBATCH --signal=B:SIGTERM@120   # 120 sec grace period for cleanup after scancel is sent
#SBATCH --mail-type=END          # choice could be 'fail'
#SBATCH --mail-user=db1224@princeton.edu

function cleanup() {
    echo 'Running cleanup script'
    kill $TRAIN_PID
    kill $TB_PID
    mv /scratch/$USER/runs/slurm_$SLURM_JOB_ID /n/fs/audiovis/$USER/DAC-JAX/runs
    rm -rf /scratch/$USER
    exit 0
}

## Trap the SIGTERM signal (sent by scancel) and call the cleanup function
trap cleanup EXIT SIGINT SIGTERM

module purge
module load anaconda3/2024.02

eval "$(conda shell.bash hook)"

conda activate ../Terrapin/.env/jax-env
export PYTHONPATH=$PWD

export DAC_JAX_CACHE=/n/fs/audiovis/$USER/DAC-JAX/model_cache
python -m dac_jax download_model --model_type=44khz --model_bitrate=8kbps

## prepare data
echo Copying data to /scratch
mkdir -p /scratch/$USER/datasets/nsynth
rsync -avH --progress /n/fs/audiovis/$USER/datasets/nsynth /scratch/$USER/datasets
echo Copied data to /scratch

## Launch TensorBoard and get the process ID of TensorBoard
tensorboard --logdir=/scratch/$USER/runs --port=10013 --samples_per_plugin audio=20 --bind_all & TB_PID=$!

python train_simple.py \
  --args.load conf/final/44khz.yml \
  --train.batch_size=12 \
  --train.val_batch_size=12 \
  --create_dataset.worker_count=2 \
  --train.name=slurm_$SLURM_JOB_ID \
  --train.ckpt_dir=/scratch/$USER/runs \
  & TRAIN_PID=$!

wait $TRAIN_PID
