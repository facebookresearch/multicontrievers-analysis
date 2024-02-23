#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

## extract vectors
python Winobias_extract_vectors.py --model contriever --seed 0

## probe and report to wandb (in summary stats)
python Winobias_MDL_probing.py --model contriever --contriever_type contriever --model_seed 0 --seed 0
