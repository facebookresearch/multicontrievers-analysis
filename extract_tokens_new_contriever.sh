#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

source activate /home/seraphina/.conda/envs/sgt

shard=$1
dataset="md_gender/wikipedia_binary" #md_gender/wizard biasbios md_gender/wizard_binary
dataset_name=wikipedia_binary #"wizard_binary"
traintext="data/${dataset}/${shard}.train.pickle"
devtext="data/${dataset}/${shard}.dev.pickle"
testtext="data/${dataset}/${shard}.test.pickle"

## extract tokens
python extract_tokens_from_dataset.py --model contriever_new --batch --datapaths ${devtext} ${testtext} ${traintext} --dataset_name $dataset_name --tensor_dataset
#--data_type scrubbed
