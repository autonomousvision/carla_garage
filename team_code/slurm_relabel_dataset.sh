#!/bin/bash
#SBATCH --job-name=re_id_000
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --nodes=1
#SBATCH --time=1-00:00
#SBATCH --gres=gpu:0
#SBATCH --mem=256G
#SBATCH --output=/path/to/%j.out  # File to which STDOUT will be written
#SBATCH --error=/path/to/%j.err   # File to which STDERR will be written
#SBATCH --partition=gpu-2080ti

# print info about current job
scontrol show job $SLURM_JOB_ID

pwd
export CARLA_ROOT=/path/to/CARLA
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/":${PYTHONPATH}
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/miniconda3/lib/

# We parallelize across processes so we don't want threading.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
srun torchrun --nnodes=1 --nproc_per_node=72 --max_restarts=0 --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d relabel_dataset.py --model_file /path/to/plant_010_1/model_0046.pth --root_dir /path/to/dataset --batch_size 10
