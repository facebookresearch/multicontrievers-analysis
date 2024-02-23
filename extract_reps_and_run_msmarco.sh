#!/bin/bash


# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

source activate /home/seraphina/.conda/envs/sgt

model_seed=$1
data_type="scrubbed" #raw is default, raw/scrubbed
ct="contriever-msmarco"

dataset=biasbios #"md_gender/wizard_binary" #"md_gender/wizard" # biasbios
dataset_name=biasinbios #"wizard_binary"  # wizard "biasinbios"
dataset_name_probe=biasinbios_profession #biasinbios #biasinbios_profession


train_tok="data/${dataset}/train.tokens_${data_type}_contriever.pt"
dev_tok="data/${dataset}/dev.tokens_${data_type}_contriever.pt"
test_tok="data/${dataset}/test.tokens_${data_type}_contriever.pt"

## extract vectors
time python extract_vectors_from_dataset.py --model contriever --batch --data_type ${data_type} \
        -ct $ct --seed ${model_seed} --datapaths ${train_tok} ${dev_tok} ${test_tok} --dataset_name ${dataset_name}

train_vec="data/${dataset}/vectors_extracted_from_trained_models/$ct/seed_${model_seed}/train.vectors_${data_type}_contriever.pt"
dev_vec="data/${dataset}/vectors_extracted_from_trained_models/$ct/seed_${model_seed}/dev.vectors_${data_type}_contriever.pt"
test_vec="data/${dataset}/vectors_extracted_from_trained_models/$ct/seed_${model_seed}/test.vectors_${data_type}_contriever.pt"

## probe and report to wandb (in summary stats)
time python run_MDL_probing_wrapper.py --model contriever --contriever_type $ct --data_type ${data_type} \
        --model_seed ${model_seed}  --seed ${model_seed} --train ${train_vec} --dev ${dev_vec} --test ${test_vec}  --dataset_name ${dataset_name_probe}
