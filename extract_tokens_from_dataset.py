# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import yaml
from transformers import RobertaTokenizer, AutoTokenizer
from utils.DataUtils import extract_tokens
from utils.ScriptUtils import data_processing_args, general_pipeline_args
from contriever.src.contriever import load_retriever


def setup_argparse():
    parser = argparse.ArgumentParser(description="Extracting tokens for training.")
    data_processing_args(parser)
    general_pipeline_args(parser)
    parser.add_argument(
        "--tensor_dataset",
        action="store_true",
        help="put tokens in tensor dataset form (more efficient), but does not contain string references for readability",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = setup_argparse()
    print(args)

    if args.model == "roberta":
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    elif args.model == "bert":
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    elif args.model == "tasb":
        tokenizer = AutoTokenizer.from_pretrained(
            "sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco"
        )
    elif args.model == "distilbert":
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    else:
        if args.model == "contriever_new":
            model_config = yaml.load(open(args.model_config), Loader=yaml.FullLoader)
            model_path = model_config["model_paths"][
                1
            ]  # all tokenizers should be the same, so we don't have to do this loads
            model, tokenizer, _ = load_retriever(model_path)
            tokenizer.model_max_length = 512  # the multiberts don't have this in their config so sequences can end up too long
        else:
            tokenizer = AutoTokenizer.from_pretrained(f"facebook/{args.model}")

    for datapath in args.datapaths:
        print(f"Working on {datapath}")
        extract_tokens(
            datapath,
            args.data_type,
            tokenizer,
            model_type=args.model,
            batch=args.batch,
            dataset=args.dataset_name,
            tensor_dataset=args.tensor_dataset,
        )
