# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import numpy as np
import torch
import wandb
from torch.utils.data import TensorDataset
from transformers import set_seed

from src.Data import BasicDataset


def parse_training_args(replace_top_layer=False):
    parser = argparse.ArgumentParser(
        description="Run finetuning training process on Bias in Bios dataset."
    )
    parser.add_argument(
        "--batch_size",
        "-b",
        default=64,
        type=int,
        help="the batch size for the training process",
    )
    parser.add_argument(
        "--data",
        "-d",
        required=False,
        type=str,
        help="the data type",
        choices=["raw", "scrubbed", "name", "scrubbed_extra"],
        default="raw",
    )
    parser.add_argument(
        "--balanced",
        type=str,
        help="balancing of the data",
        choices=["subsampled", "oversampled", "original"],
        default="original",
    )
    parser.add_argument("--lr", default=5e-5, type=float, help="the learning rate")
    parser.add_argument(
        "--epochs", "-e", default=10, type=int, help="the number of epochs"
    )
    parser.add_argument("--seed", "-s", default=0, type=int, help="the random seed")
    parser.add_argument(
        "--embedding_seed",
        default=None,
        type=int,
        help="the random seed embedding was trained with",
    )
    parser.add_argument(
        "--printevery",
        "-pe",
        default=1,
        type=int,
        help="print results every this number of epochs",
    )
    parser.add_argument(
        "--checkpointevery",
        "-ce",
        default=1,
        type=int,
        help="print results every this number of epochs",
    )

    parser.add_argument(
        "--poe", action="store_true", help="whether to use product of experts training"
    )
    parser.add_argument(
        "--poe_hard",
        action="store_true",
        help="whether to use product of experts training hard version",
    )
    parser.add_argument(
        "--bias_only",
        required=False,
        type=str,
        help="the type of bias-only branch in poe",
        choices=["scrubbed", "name", "scrubbed_words"],
    )
    parser.add_argument(
        "--dfl", action="store_true", help="whether to use debiased focal loss training"
    )
    parser.add_argument(
        "--dfl_hard",
        action="store_true",
        help="whether to use debiased focal loss training hard version",
    )
    parser.add_argument(
        "--adam_epsilon", default=1e-6, type=float, help="Epsilon for Adam optimizer."
    )

    if replace_top_layer:
        parser.add_argument(
            "--embedding_training_data",
            required=False,
            type=str,
            help="the data type of pretrained embeddings",
            choices=["raw", "scrubbed", "name"],
            default="raw",
        )
        parser.add_argument(
            "--embedding_training_balanced",
            type=str,
            help="balancing of the data for pretrained embeddings",
            choices=["subsampled", "oversampled", "original"],
            default="original",
        )

    args = parser.parse_args()

    print("Batch size:", args.batch_size)
    print("Data type:", args.data)
    print("Balancing:", args.balanced)
    print("Learning rate:", args.lr)
    print("Number of epochs:", args.epochs)
    print("Random seed:", args.seed)
    print("Print Every:", args.printevery)
    print("Checkpoint Every:", args.checkpointevery)

    return args


def parse_testing_args():
    parser = argparse.ArgumentParser(description="Run testing on Bias in Bios dataset.")
    parser.add_argument(
        "--batch_size", "-b", default=64, type=int, help="the batch size to test with"
    )
    parser.add_argument(
        "--training_data",
        required=False,
        type=str,
        help="the data type the model was trained on",
        choices=["raw", "scrubbed", "name", "scrubbed_extra"],
        default="raw",
    )
    parser.add_argument(
        "--testing_data",
        required=False,
        type=str,
        help="the data type to test on",
        choices=["raw", "scrubbed", "name", "scrubbed_extra"],
        default="raw",
    )
    parser.add_argument(
        "--training_balanced",
        type=str,
        help="balancing of the training data",
        choices=["subsampled", "oversampled", "original"],
        default="original",
    )
    parser.add_argument(
        "--testing_balanced",
        type=str,
        help="balancing of the test data",
        choices=["subsampled", "oversampled", "original"],
        default="original",
    )
    parser.add_argument(
        "--split",
        type=str,
        help="the split type to test",
        choices=["train", "test", "valid"],
        default="test",
    )
    parser.add_argument("--seed", "-s", default=0, type=int, help="the random seed")
    parser.add_argument(
        "--poe", action="store_true", help="whether to use product of experts training"
    )
    parser.add_argument(
        "--poe_hard",
        action="store_true",
        help="whether to use product of experts training hard version",
    )
    parser.add_argument(
        "--bias_only",
        required=False,
        type=str,
        help="the type of bias-only branch in poe",
        choices=["scrubbed", "name", "scrubbed_words"],
    )
    parser.add_argument(
        "--dfl", action="store_true", help="whether to use debiased focal loss training"
    )
    parser.add_argument(
        "--dfl_hard",
        action="store_true",
        help="whether to use debiased focal loss training hard version",
    )

    args = parser.parse_args()

    print("Batch size:", args.batch_size)
    print("Training data type:", args.training_data)
    print("Testing data type:", args.testing_data)
    print("Training Balancing:", args.training_balanced)
    print("Testing Balancing:", args.testing_balanced)
    print("Split Type:", args.split)
    print("Random seed:", args.seed)

    return args


def general_pipeline_args(parser):
    parser.add_argument(
        "--model",
        required=True,
        type=str,
        help="the model type",
        choices=[
            "basic",
            "finetuning",
            "LM",
            "random",
            "coref",
            "contriever",
            "contriever_new",
            "bert",
            "bm25",
            "tasb",
            "distilbert",
        ],
    )
    parser.add_argument(
        "--contriever_type",
        "-ct",
        type=str,
        choices=[
            "contriever",
            "contriever-msmarco",
            "mcontriever",
            "mcontriever-msmarco",
        ],
        default="contriever",
    )
    parser.add_argument(
        "--data_type",
        type=str,
        help="the type of data used (raw or modified)",
        choices=["raw", "scrubbed", "scrubbed_extra", "inlp"],
        default="raw",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        choices=[
            "biasinbios",
            "biasinbios_profession",
            "wizard",
            "wizard_binary",
            "wikipedia",
            "wikipedia_binary",
        ],
        default="biasinbios",
    )
    return parser


def data_processing_args(parser):
    parser.add_argument(
        "--batch", action="store_true", help="whether to use batched tensors"
    )
    parser.add_argument(
        "--datapaths", nargs="+", help="datapaths of files to extract tokens"
    )
    parser.add_argument(
        "--model_config",
        default="config/contrievers.yaml",
        help="config of model paths, by seed",
    )
    return parser


def load_tensor_dataset(args):
    # tensor datasets of X, z. Assumes each is a TensorDataset object
    all_labels = set()
    all_ds = []
    for ds_name in [args.train, args.dev, args.test]:
        ds = torch.load(ds_name, map_location="cuda")
        all_labels = all_labels | set(i.item() for i in ds.tensors[1].unique())
        all_ds.append(ds)
    n_labels = len(all_labels)
    # TODO implement no_dev option

    return *all_ds, n_labels


def load_bias_in_bios_vectors(args):
    if args.dataset_name == "biasinbios_profession":
        ds = BasicDataset(args.train, args.dev, args.test, label_key="y")
    else:
        ds = BasicDataset(args.train, args.dev, args.test)
    ds.preprocess_probing_data()

    return ds.train, ds.dev, ds.test, ds.n_labels


def load_wizard_vectors(args):
    ds = BasicDataset(args.train, args.dev, args.test, no_dev=True)
    ds.preprocess_probing_data()
    # since it is small, combine dev and test
    return ds.train, ds.dev, ds.test, ds.n_labels


def load_winobias_vectors(args, preprocess=True):
    if preprocess:
        load_fn = load_probing_dataset
    else:
        load_fn = torch.load

    if args.training_task == "coref" and args.model == "finetuned":
        data = load_fn(
            f"data/winobias/extracted_vectors/roberta-base/finetuned/{args.training_balanced}/number_{args.model_number}/vectors_roberta-base.pt"
        )
    if args.training_task == "biasinbios" and args.model == "finetuned":
        data = load_fn(
            f"data/winobias/extracted_vectors/roberta-base/finetuned/bios/{args.training_data}/{args.training_balanced}/seed_{args.model_seed}/vectors_roberta-base.pt"
        )
    if args.model == "random":
        data = load_fn(
            f"data/winobias/extracted_vectors/roberta-base/random/seed_{args.seed}/vectors_roberta-base.pt"
        )
    if args.model == "basic":
        data = load_fn(
            f"data/winobias/extracted_vectors/roberta-base/basic/vectors_roberta-base.pt"
        )

    return data, None, data  # TODO why are these the same?


def get_avg_gap(gap):
    gap = np.array(gap)
    f = np.mean(gap[gap > 0])
    m = -np.mean(gap[gap < 0])
    return {"f": f, "m": m}


def get_gap_sum(gap):
    return np.abs(np.array(gap)).sum()


def log_test_results(res):
    wandb.run.summary[f"acc"] = res["acc"]
    wandb.run.summary[f"avg_loss"] = res["loss"]

    perc = res["perc"]

    # gaps
    wandb.run.summary[f"tpr_gap-pearson"] = res["pearson_tpr_gap"]
    # wandb.run.summary[f"tpr_gap_avg_per_gender"] = get_avg_gap(res['tpr_gap'])
    wandb.run.summary[f"tpr_gap-abs_sum"] = get_gap_sum(res["tpr_gap"])
    table_data = [[x, y] for (x, y) in zip(perc, res["tpr_gap"])]
    table = wandb.Table(data=table_data, columns=["perc of females", "tpr gap"])
    wandb.log(
        {
            f"tpr_gap_chart": wandb.plot.line(
                table, "perc of females", "tpr gap", title=f"tpr gap chart"
            )
        }
    )

    wandb.run.summary[f"fpr_gap-pearson"] = res["pearson_fpr_gap"]
    # wandb.run.summary[f"fpr_gap_avg_per_gender"] = get_avg_gap(res['fpr_gap'])
    wandb.run.summary[f"fpr_gap-abs_sum"] = get_gap_sum(res["fpr_gap"])
    table_data = [[x, y] for (x, y) in zip(perc, res["fpr_gap"])]
    table = wandb.Table(data=table_data, columns=["perc of females", "fpr gap"])
    wandb.log(
        {
            f"fpr_gap_chart": wandb.plot.line(
                table, "perc of females", "fpr gap", title=f"fpr gap chart"
            )
        }
    )

    wandb.run.summary[f"precision_gap-pearson"] = res["pearson_precision_gap"]
    # wandb.run.summary[f"precision_gap_avg_per_gender"] = get_avg_gap(res['precision_gap'])
    wandb.run.summary[f"precision_gap-abs_sum"] = get_gap_sum(res["precision_gap"])
    table_data = [[x, y] for (x, y) in zip(perc, res["precision_gap"])]
    table = wandb.Table(data=table_data, columns=["perc of females", "precision gap"])
    wandb.log(
        {
            f"precision_gap_chart": wandb.plot.line(
                table, "perc of females", "precision gap", title=f"precision gap chart"
            )
        }
    )

    # Allennlp metrics

    ## independence
    wandb.run.summary["independence"] = res["independence"]
    wandb.run.summary["independence-sum"] = res["independence_sum"]

    ## separation
    wandb.run.summary["separation"] = res["separation"]
    wandb.run.summary["separation_gap-abs_sum"] = get_gap_sum(res["separation_gaps"])
    # wandb.run.summary['separation_gap-pearson'] = res['pearson_separation_gaps']
    # table_data = [[x, y] for (x, y) in zip(perc, res['separation_gaps'])]
    # table = wandb.Table(data=table_data, columns=["perc of females", "separation gap"])
    # wandb.log({f"separation_gap_chart": wandb.plot.line(table, "perc of females", "separation gap",
    #                                                 title=f"separation gap chart")})

    ## sufficiency
    wandb.run.summary["sufficiency"] = res["sufficiency"]
    wandb.run.summary["sufficiency_gap-abs_sum"] = get_gap_sum(res["sufficiency_gaps"])
    # wandb.run.summary['sufficiency_gap-pearson'] = res['pearson_sufficiency_gaps']
    # table_data = [[x, y] for (x, y) in zip(perc, res['sufficiency_gaps'])]
    # table = wandb.Table(data=table_data, columns=["perc of females", "sufficiency gap"])
    # wandb.log({f"sufficiency_gap_chart": wandb.plot.line(table, "perc of females", "separation gap",
    #                                                 title=f"sufficiency gap chart")})
