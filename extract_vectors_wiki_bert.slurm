#!/bin/sh

#SBATCH --job-name=extract_vectors
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task 10
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=500

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

source /home/seraphina/.bashrc

source activate /home/seraphina/.conda/envs/sgt


model_seed=$1
data_shard=$2

data_type="raw" # raw is default, options raw, scrubbed
model="bert"
model_type="google_multiberts-seed_${model_seed}" 


dataset="md_gender/wikipedia_binary"
dataset_name=wikipedia_binary

#train_tok="data/${dataset}/${data_shard}.train.tokens_${data_type}_${model}.pt"
#dev_tok="data/${dataset}/${data_shard}.dev.tokens_${data_type}_${model}.pt"
test_tok="data/${dataset}/${data_shard}.test.tokens_${data_type}_${model}.pt"

echo "Running script"

srun /home/seraphina/.conda/envs/sgt/bin/python extract_vectors_from_dataset.py --model ${model} --data_type ${data_type} --batch --tensor_dataset \
                        --seed ${model_seed} --datapaths ${test_tok} --dataset_name ${dataset_name}

echo "Done"
#${train_tok} ${dev_tok} 
