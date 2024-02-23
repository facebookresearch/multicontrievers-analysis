#!/bin/sh

#SBATCH --job-name=extract_tokens
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task 10
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=180

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

source /home/seraphina/.bashrc

source activate /home/seraphina/.conda/envs/sgt

script=$1
shard=$2

echo "Running script ${script}"

bash ${script} ${shard}

echo "Done"
