#!/bin/bash
#SBATCH --job-name=train_id_000
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=3-00:00
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --output=/path/to/slurmlogs/%j.out  # File to which STDOUT will be written
#SBATCH --error=/path/to/slurmlogs/%j.err   # File to which STDERR will be written
#SBATCH --partition=a100

# print info about current job
scontrol show job $SLURM_JOB_ID

pwd
export CARLA_ROOT=/path/to/carla_9_10
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/":${PYTHONPATH}
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/miniconda/lib

export OMP_NUM_THREADS=32  # Limits pytorch to spawn at most num cpus cores threads
export OPENBLAS_NUM_THREADS=1  # Shuts off numpy multithreading, to avoid threads spawning other threads.
torchrun --nnodes=1 --nproc_per_node=8 --max_restarts=0 --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d train.py --id train_id_000 --batch_size 8 --setting 02_05_withheld --root_dir /path/to/dataset --logdir /path/to/logdir --use_controller_input_prediction 1 --use_wp_gru 0 --use_discrete_command 1 --use_tp 1 --continue_epoch 1 --cpu_cores 32 --num_repetitions 3
