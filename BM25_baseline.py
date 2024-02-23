# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import pickle
import os
import re
from functools import partial
from pathlib import Path
import spacy
import torch
import numpy as np

from src.BM25 import BM25
from utils.ScriptUtils import data_processing_args, general_pipeline_args
from utils.DataUtils import process_dataset, dataset2filebase


def spacy_tokenize(text, nlp):
    return [x.orth_ for x in nlp(text)]


def setup_argparse():
    parser = argparse.ArgumentParser(description="fitting IDF to corpus.")
    data_processing_args(parser)
    general_pipeline_args(parser)
    parser.add_argument("--fit", action="store_true")
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


if __name__ == "__main__":
    args = setup_argparse()
    print(args)

    model_path = f"baseline/{args.dataset_name}_{args.data_type}_{args.model}_model.pkl"

    if args.fit:
        nlp = spacy.load("en_core_web_lg", disable=["ner", "parser", "tagger"])
        tokenizer = partial(spacy_tokenize, nlp=nlp)

        bm25 = BM25(tokenizer=tokenizer)
        # need a corpus to fit to
        corpus = []
        for datapath in args.datapaths:
            with open(datapath, "rb") as f:
                ds = pickle.load(f)

            print(f"Working on {datapath}")
            outfile = f"{os.path.splitext(datapath)[0]}.tokens_{args.data_type}_{args.model}.pt"
            inputs, labels, genders = process_dataset(
                args.dataset_name, ds, args.data_type
            )

            corpus.extend(inputs)
            outdict = {"X": inputs, "y": labels, "z": genders}
            torch.save(outdict, outfile)

        print("Fitting to corpus")
        bm25.fit(corpus)

        model_out = (
            f"baseline/{args.dataset_name}_{args.data_type}_{args.model}_model.pkl"
        )
        with open(model_path, "wb") as fout:
            pickle.dump(bm25, fout)

    if args.extract:
        with open(model_path, "rb") as fin:
            model = pickle.load(fin)
        filebase = dataset2filebase[args.dataset_name]
        output_folder = os.path.join(filebase, f"{args.model}/seed_{args.seed}/")
        Path(output_folder).mkdir(parents=True, exist_ok=True)

        for datapath in args.datapaths:
            print(f"Working on {datapath}")
            filename = os.path.split(datapath)[1]
            filename = re.sub("tokens", "vectors", filename)
            data = torch.load(datapath)
            vectors = model.vectorizer.transform(data["X"])

            outfile = os.path.join(output_folder, filename)
            print(f"Saving to {outfile}")
            labels, genders = np.array(data["y"]), np.array(data["z"])
            torch.save({"X": vectors, "y": labels, "z": genders}, outfile)
