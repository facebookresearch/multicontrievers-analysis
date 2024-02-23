# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import random
import os
import pickle
import yaml

import torch
from transformers import (
    RobertaModel,
    RobertaForMaskedLM,
    RobertaConfig,
    AutoModel,
    BertModel,
)
from transformers import RobertaTokenizer
import numpy as np
from utils.DataUtils import extract_vectors, dataset2filebase
from utils.ScriptUtils import data_processing_args, general_pipeline_args
from src.Models import roBERTa_classifier
from src.Trainer import load_checkpoint
from contriever.src.contriever import Contriever, load_retriever

N_LABELS = 28


def setup_argparse():
    parser = argparse.ArgumentParser(
        description="Extracting vectors from trained Bias in Bios model."
    )
    data_processing_args(parser)
    general_pipeline_args(parser)
    parser.add_argument(
        "--seed",
        "-s",
        default=42,
        type=int,
        help="the random seed the model was trained on",
    )
    parser.add_argument(
        "--tensor_dataset",
        action="store_true",
        help="tokens already in tensor dataset form (more efficient)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = setup_argparse()
    print("model:", args.model)
    print("seed:", args.seed)
    print(args)

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    filebase = dataset2filebase[args.dataset_name]

    if args.model in ["contriever", "contriever_new"]:
        if args.model == "contriever":
            model = Contriever.from_pretrained(f"facebook/{args.contriever_type}")
        else:
            model_config = yaml.load(open(args.model_config), Loader=yaml.FullLoader)
            model_path = model_config["model_paths"][args.seed]
            model, tokenizer, _ = load_retriever(model_path)
        ouput_folder = os.path.join(
            filebase, f"{args.contriever_type}/seed_{args.seed}/"
        )
    elif args.model == "bert":
        if seed == 42:
            model_version = "bert-base-uncased"
            model = BertModel.from_pretrained(model_version)
        else:
            model_version = f"google/multiberts-seed_{args.seed}"
        model = BertModel.from_pretrained(model_version)
        model_version = model_version.replace("/", "_")
        ouput_folder = os.path.join(filebase, f"{model_version}/seed_{args.seed}/")
    elif args.model == "tasb":
        model = AutoModel.from_pretrained(
            "sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco"
        )
        ouput_folder = os.path.join(filebase, f"{args.model}/seed_{args.seed}/")
    elif args.model == "distilbert":
        model = AutoModel.from_pretrained("distilbert-base-uncased")
        ouput_folder = os.path.join(filebase, f"{args.model}/seed_{args.seed}/")
    else:
        model_version = "roberta-base"
        if args.model == "basic":
            model = RobertaModel.from_pretrained(
                model_version, output_attentions=True, output_hidden_states=True
            )
            ouput_folder = "data/biasbios/"
        if args.model == "random":
            configuration = RobertaConfig()
            model = AutoModel.from_config(configuration)
            tokenizer = RobertaTokenizer.from_pretrained(model_version)
            model.resize_token_embeddings(len(tokenizer))
            ouput_folder = f"roberta-base/random/seed_{args.seed}"
        elif args.model == "finetuning":
            load_path = f"checkpoints/bias_in_bios/roberta-base/finetuning/{args.training_data}/{args.training_balanced}/seed_{args.seed}/best_model.pt"
            model_ = roBERTa_classifier(N_LABELS)
            load_checkpoint(model_, load_path)
            model = model_.roberta
            ouput_folder = os.path.join(
                filebase,
                f"roberta-base/{args.model}/{args.training_data}/{args.training_balanced}/seed_{args.seed}",
            )
        elif args.model == "LM":
            load_path = f"checkpoints/bias_in_bios/roberta-base/LM/{args.training_data}/{args.training_balanced}/seed_{args.seed}/best_model.pt"
            model_ = RobertaForMaskedLM.from_pretrained("roberta-base")
            load_checkpoint(model_, load_path)
            model = model_.roberta
            ouput_folder = os.path.join(
                filebase,
                f"/roberta-base/{args.model}/{args.training_data}/{args.training_balanced}/seed_{args.seed}",
            )

    for datapath in args.datapaths:
        print(f"Working on {datapath}")
        extract_vectors(
            args.data_type,
            model,
            ouput_folder,
            data_path=datapath,
            batch=args.batch,
            tensor_dataset=args.tensor_dataset,
        )
