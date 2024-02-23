#!/bin/sh

#SBATCH --job-name=inlp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task 10
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=480

source /home/seraphina/.bashrc

source activate /home/seraphina/.conda/envs/sgt

script=$1
model=$2
seed=$3

echo "Running script ${script}"

bash ${script} ${model} ${seed} 

echo "Done"
