# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
from dataclasses import dataclass
from os.path import join
from typing import List

import torch
import wandb
from torch import nn
import sys

from transformers import set_seed

from probing.MDLProbingUtils import (
    build_probe,
    OnlineCodingExperimentResults,
    create_probe,
    run_MDL_probing,
    general_MDL_args,
)

from utils.ScriptUtils import load_winobias_vectors


def parse_args():
    parser = general_MDL_args()
    parser.add_argument(
        "--model",
        required=True,
        type=str,
        help="the model type",
        choices=["basic", "finetuned", "random", "contriever"],
    )
    parser.add_argument(
        "--training_task",
        default="coref",
        required=False,
        type=str,
        help="the model original training task",
        choices=["coref", "biasinbios"],
    )
    parser.add_argument(
        "--training_balanced",
        type=str,
        help="balancing of the training data",
        choices=[
            "balanced",
            "imbalanced",
            "anon",
            "CA",
            "subsampled",
            "oversampled",
            "original",
        ],
        default=None,
    )
    parser.add_argument(
        "--training_data",
        type=str,
        help="data type of the training data, for biasbios task",
        choices=["raw", "scrubbed"],
        default=None,
    )
    parser.add_argument(
        "--model_number",
        "-n",
        type=int,
        help="the model number to check on",
        required=False,
    )
    parser.add_argument(
        "--contriever_type",
        "-ct",
        type=str,
        choices=["contriever", "msmarco", "m-contriever", "m-msmarco"],
        default="contriever",
    )

    args = parser.parse_args()
    print(args)
    return args


def init_wandb(args):
    wandb.init(
        project="Winobias MDL probing",
        config={
            "architecture": "RoBERTa",
            "seed": args.seed,
            "training balancing": args.training_balanced,
            "training_data": args.training_data,
            "model": args.model,
            "model_number": args.model_number,
            "model_seed": args.model_seed,
            "training_task": args.training_task,
        },
    )


def main():
    args = parse_args()
    init_wandb(args)

    task_name = f"winobias_model_{args.training_task}_{args.model}_training_{args.training_balanced}_seed_{args.seed}_number_{args.model_number}"
    results = run_MDL_probing(
        args, load_winobias_vectors, task_name, shuffle=False
    )  # returns compression numbers but then does nothing with them :p

    print(results)


if __name__ == "__main__":
    main()
