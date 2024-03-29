#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

traintext="data/biasbios/train.pickle"
devtext="data/biasbios/dev.pickle"
testtext="data/biasbios/test.pickle"

## extract tokens
python  --model contriever --batch --datapaths ${traintext} ${devtext} ${testtext}

train_tok="data/biasbios/train.tokens_raw_contriever.pt"
dev_tok="data/biasbios/dev.tokens_raw_contriever.pt"
test_tok="data/biasbios/test.tokens_raw_contriever.pt"

## extract vectors
python extract_vectors_from_dataset.py --model contriever --batch -ct contriever --datapaths  ${traintext} ${devtext} ${testtext}  

train_vec="data/biasbios/vectors_extracted_from_trained_models/contriever/seed_0/train.vectors_raw_contriever.pt"
dev_vec="data/biasbios/vectors_extracted_from_trained_models/contriever/seed_0/dev.vectors_raw_contriever.pt"
test_vec="data/biasbios/vectors_extracted_from_trained_models/contriever/seed_0/test.vectors_raw_contriever.pt"

## probe and report to wandb (in summary stats)
python run_MDL_probing_wrapper.py --model contriever --contriever_type contriever --model_seed 0 --seed 0 \
                                --train ${train_vec} --dev ${dev_vec} --test ${test_vec}
