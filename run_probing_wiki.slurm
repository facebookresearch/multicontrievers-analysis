#!/bin/sh

#SBATCH --job-name=run_probing
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


model_seed=$1
dataset_shard=$2

data_type="raw" # raw is default, options raw, scrubbed
# model="contriever_new"
# model_type="contriever"

model="bert"
model_type="google_multiberts-seed_${model_seed}" 


dataset="md_gender/wikipedia_binary"
dataset_name=wikipedia_binary

train_vec="data/${dataset}/vectors_extracted_from_trained_models/${model_type}/seed_${model_seed}/${dataset_shard}.train.vectors_${data_type}_${model}.pt"
dev_vec="data/${dataset}/vectors_extracted_from_trained_models/${model_type}/seed_${model_seed}/${dataset_shard}.dev.vectors_${data_type}_${model}.pt"
test_vec="data/${dataset}/vectors_extracted_from_trained_models/${model_type}/seed_${model_seed}/${dataset_shard}.test.vectors_${data_type}_${model}.pt"

echo "Running script with model seed ${model_seed} and dataset shard ${dataset_shard}"

srun /home/seraphina/.conda/envs/sgt/bin/python run_MDL_probing_wrapper.py --model ${model}  --model_seed ${model_seed} --seed ${model_seed} --data_type ${data_type} \
                                --train ${train_vec} --dev ${dev_vec} --test ${test_vec} --dataset_name ${dataset_name} --tensor_dataset --dataset_shard ${dataset_shard}

echo "Done"
