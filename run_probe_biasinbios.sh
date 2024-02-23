#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

source activate /home/seraphina/.conda/envs/sgt

traintext="data/biasbios/train.pickle"
devtext="data/biasbios/dev.pickle"
testtext="data/biasbios/test.pickle"

model=bert
model_type="bert-base-uncased" #contriever-msmarco contriever mcontriever mcontriever-msmarco
## extract tokens
#python extract_tokens_from_dataset.py --model ${model}--datapaths ${traintext} ${devtext} ${testtext}

train_tok="data/biasbios/train.tokens_raw_${model}_unbatched.pt"
dev_tok="data/biasbios/dev.tokens_raw_${model}_unbatched.pt"
test_tok="data/biasbios/test.tokens_raw_${model}_unbatched.pt"

## extract vectors
#python extract_vectors_from_dataset.py --model ${model} --datapaths ${dev_tok} ${test_tok} ${train_tok}
#python extract_vectors_from_dataset.py --model ${model} --batch -ct contriever-msmarco --datapaths  ${train_tok} ${dev_tok} ${test_tok}  

train_vec="data/biasbios/vectors_extracted_from_trained_models/${model_type}/seed_0/train.vectors_raw_${model}_unbatched.pt"
dev_vec="data/biasbios/vectors_extracted_from_trained_models/${model_type}/seed_0/dev.vectors_raw_${model}_unbatched.pt"
test_vec="data/biasbios/vectors_extracted_from_trained_models/${model_type}/seed_0/test.vectors_raw_${model}_unbatched.pt"

## probe and report to wandb (in summary stats)
python run_MDL_probing_wrapper.py --model ${model}  --model_seed 0 --seed 0 \
                                --train ${train_vec} --dev ${dev_vec} --test ${test_vec}
# python run_MDL_probing_wrapper.py --model contriever --contriever_type contriever-msmarco --model_seed 0 --seed 0 \
#                                 --train ${train_vec} --dev ${dev_vec} --test ${test_vec}
