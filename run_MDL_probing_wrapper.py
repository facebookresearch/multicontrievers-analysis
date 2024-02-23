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

# sys.path.append('../src')
from probing.mdl import OnlineCodeMDLProbe
from utils.ScriptUtils import (
    load_bias_in_bios_vectors,
    load_wizard_vectors,
    general_pipeline_args,
    load_tensor_dataset,
)


dataset2loadfunc = {
    "biasinbios": load_bias_in_bios_vectors,
    "biasinbios_profession": load_bias_in_bios_vectors,
    "wizard": load_wizard_vectors,
    "wizard_binary": load_wizard_vectors,
}


def parse_args():
    parser = general_MDL_args()
    general_pipeline_args(parser)
    parser.add_argument(
        "--tensor_dataset",
        action="store_true",
        help="tokens already in tensor dataset form (more efficient)",
    )
    parser.add_argument("--dataset_shard", type=int)
    parser.add_argument("--train", type=str)
    parser.add_argument("--dev", type=str)
    parser.add_argument("--test", type=str)

    args = parser.parse_args()
    print(args)
    return args


def init_wandb(args):
    config_dict = {
        "model": args.model,
        "model_subtype": args.contriever_type
        if args.model == "contriever"
        else args.model,
        "seed": args.seed,
        "model_seed": args.model_seed,
        "type": args.data_type,
    }
    if args.dataset_shard:
        config_dict["dataset_shard"] = args.dataset_shard
    wandb.init(project=f"{args.dataset_name} MDL probing", config=config_dict)


def main():
    args = parse_args()
    init_wandb(args)
    if args.model == "contriever":
        task_name = f"biasinbios_model_{args.contriever_type}_type_{args.data_type}_seed_{args.seed}"
    else:
        task_name = (
            f"biasinbios_model_{args.model}_type_{args.data_type}_seed_{args.seed}"
        )

    load_fn = (
        load_tensor_dataset
        if args.tensor_dataset
        else dataset2loadfunc[args.dataset_name]
    )
    run_MDL_probing(args, load_fn, task_name, shuffle=True)


if __name__ == "__main__":
    main()
