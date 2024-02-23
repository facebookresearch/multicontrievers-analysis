# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Graph results either by a scatterplot for correlation of compression and performance or a distribution over performances. 
Example usage:
python graph_results.py --performance_only --out_dir graphs/perf_dist/gender 
--data_path /home/seraphina/contriever/beir/results/contriever/all_metrics_all_seeds_gender.pkl --by_dataset
"""

import argparse
import os

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

import ipdb


def get_reference_performance():
    reference_dict = {}
    reference_dict["arguana"] = 37.9
    reference_dict["climate-fever"] = 15.5
    reference_dict["fiqa"] = 24.5
    reference_dict["nfcorpus"] = 31.7
    reference_dict["scidocs"] = 14.9
    reference_dict["scifact"] = 64.9
    reference_dict["trec-covid-beir"] = 27.4
    reference_dict["webis-touche2020"] = 19.3
    reference_dict["dbpedia-entity"] = 29.2
    reference_dict["fever"] = 68.2
    reference_dict["hotpotqa"] = 48.1
    reference_dict["msmarco"] = 20.6
    reference_dict["nq"] = 25.4
    reference_dict["quora"] = 83.5
    reference_dict["all"] = 37.1
    return reference_dict


def plot_and_save(df, metric, outfile, plot_type="scatter", legend=True):
    if plot_type == "scatter":
        ax = sns.scatterplot(
            data=df,
            x="compression",
            y="value",
            hue="seed",
            legend=legend,
            palette="colorblind",
            s=100,
        )
    elif plot_type == "reg":
        ax = sns.regplot(data=df, x="compression", y="value")
    elif plot_type == "box":
        ax = sns.boxplot(data=df, x="compression", y="value", hue="seed", legend=legend)
    # ipdb.set_trace()
    if legend:
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    plt.ylabel(metric)
    plt.savefig(outfile, bbox_inches="tight")
    plt.clf()


def setup_argparse():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data_path",
        help="dataframe to load",
        default="results/all_metrics_all_seeds.pkl",
    )
    p.add_argument("--by_dataset", action="store_true")
    p.add_argument("--metric", type=str, default="NDCG@10")
    p.add_argument("--model", type=str, default="contriever")
    p.add_argument("--out_dir", type=str, default="graphs/")
    p.add_argument(
        "--plot_type", type=str, default="scatter", choices=["scatter", "box", "reg"]
    )
    p.add_argument("--data_type", type=str, choices=["raw", "scrubbed"], default="raw")
    p.add_argument("--performance_only", action="store_true")
    p.add_argument("--no_legend", action="store_false")
    return p.parse_args()


if __name__ == "__main__":
    args = setup_argparse()
    print(args)

    df = pd.read_pickle(args.data_path)

    # any custom excludes
    df = df[df["seed"] != "seed_13"]  # this seed currently broken

    df = df[df["model"] == args.model]
    df = df[df["data_type"] == args.data_type]
    metric_df = df[df["metric"] == args.metric]

    if not args.by_dataset:
        df = metric_df[metric_df["dataset"] == "all"]
        if args.performance_only:
            ds_n = "all"
            reference_dict = get_reference_performance()
            myplot = sns.histplot(
                data=df, x="value", bins=10, color="silver", linewidth=0
            )
            # display settings
            myplot.set(ylim=(0, 12))
            if reference_dict.get(ds_n):
                plt.axvline(reference_dict[ds_n], color="black", linestyle="--")
            plt.ylabel("# checkpoints")
            plt.xlabel(f"{ds_n} {args.metric}")
            plt.grid()
            plt.savefig(os.path.join(args.out_dir, f"{ds_n}_performance_dist.pdf"))
            plt.clf()
        elif args.plot_type == "scatter":
            filepath = os.path.join(
                args.out_dir,
                f"{args.metric}_avg_{args.plot_type}_{args.data_type}_{args.plot_type}.pdf",
            )
            plot_and_save(
                df, args.metric, filepath, args.plot_type, legend=args.no_legend
            )
    else:
        all_ds_names = metric_df["dataset"].unique()
        for ds_n in all_ds_names:
            if ds_n == "all":
                continue
            this_ds = metric_df[metric_df["dataset"] == ds_n]
            if not args.performance_only:
                filepath = os.path.join(
                    args.out_dir,
                    f"{args.metric}_{ds_n}_{args.data_type}_{args.plot_type}.pdf",
                )
                plot_and_save(
                    this_ds,
                    args.metric,
                    filepath,
                    args.plot_type,
                    legend=args.no_legend,
                )
            else:
                reference_dict = get_reference_performance()
                myplot = sns.histplot(
                    data=this_ds, x="value", bins=10, color="silver", linewidth=0
                )
                # display settings
                myplot.set(ylim=(0, 12))
                if reference_dict.get(ds_n):
                    plt.axvline(reference_dict[ds_n], color="black", linestyle="--")
                plt.ylabel("# checkpoints")
                plt.xlabel(f"{ds_n} {args.metric}")
                plt.grid()
                plt.savefig(os.path.join(args.out_dir, f"{ds_n}_performance_dist.pdf"))
                plt.clf()
