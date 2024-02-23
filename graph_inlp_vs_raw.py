"""
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""

import argparse
import os

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def setup_argparse():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data_path",
        help="dataframe to load",
        default="results/all_metrics_all_seeds.pkl",
    )
    p.add_argument(
        "--gender_data_path",
        help="gender dataframe to load",
        default="results/all_metrics_all_seeds_gender.pkl",
    )
    p.add_argument(
        "--inlp_data_path", help="dataframe to load", default="results/inlp_17.pkl"
    )
    p.add_argument(
        "--inlp_gender_data_path",
        help="dataframe to load",
        default="results/inlp_17_gender.pkl",
    )
    p.add_argument("--male_vs_female", action="store_true")
    # p.add_argument('--metric', type=str, default="NDCG@10")
    # p.add_argument('--model', type=str, default="contriever")
    # p.add_argument('--out_dir', type=str, default="graphs/")
    # p.add_argument('--plot_type', type=str, default="scatter", choices=["scatter", "box"])
    # p.add_argument('--data_type', type=str, choices=["raw", "scrubbed"], default="raw")
    # p.add_argument('--performance_only', action='store_true')
    return p.parse_args()


if __name__ == "__main__":
    args = setup_argparse()
    print(args)

    # combine gender_df and not
    df = pd.read_pickle(args.data_path)
    gender_df = pd.read_pickle(args.gender_data_path)
    df = pd.concat([df, gender_df], ignore_index=True)
    df["beir_experiment"] = "raw"

    inlp_df = pd.read_pickle(args.inlp_data_path)
    inlp_gender_df = pd.read_pickle(args.inlp_gender_data_path)
    inlp_df = pd.concat([inlp_df, inlp_gender_df], ignore_index=True)
    inlp_df["beir_experiment"] = "inlp"

    master_df = pd.concat([df, inlp_df])
    master_df.drop_duplicates(inplace=True)

    metrics = ["NDCG@10", "NDCG@100", "Recall@10", "Recall@100"]
    if args.male_vs_female:
        pass
        # dataset_male, dataset_female = 'nq-train-new-male', 'nq-train-new-female'
    else:
        metric_df = master_df[master_df["metric"].isin(metrics)]
        # exclusions
        mask = metric_df["dataset"].isin(["trec-covid", "all", "trec-covid-v2"])
        this_df = metric_df[~mask]
        mask = this_df["seed"] == "seed_13"

        this_df = this_df[this_df["data_type"].isin(["raw"])]
        this_df = this_df[this_df["seed"] == "seed_17"]

        myplot = sns.catplot(
            data=this_df,
            x="metric",
            hue="beir_experiment",
            y="value",
            col="dataset",
            kind="bar",
            col_wrap=5,
            sharey=False,
        )
        for ax in myplot.axes.ravel():
            # add annotations
            for i, c in enumerate(
                ax.containers
            ):  # first does one type of x then the other, then moves to next facet
                labels = [f"{v.get_height()}" for v in c]
                ax.bar_label(c, labels=labels, label_type="edge")
                vals = [v.get_height() for v in c]
                if i % 2 == 0:
                    next_vals = als = [v.get_height() for v in ax.containers[i + 1]]
                    diffs = [np.round(i - j, 2) for i, j in zip(vals, next_vals)]
                    # [np.round(vals[i] - vals[i+1], 2) for i in range(0, len(vals)-1, 2)]
                    # print(vals)
                    # print(diffs)
                    patches = []
                    for i, d in enumerate(diffs):
                        color = "green" if d < 0 else "red"

                        patch = mpatches.Patch(
                            (0.5 + 0.2 * i, 0.9),
                            color=color,
                            label=str(d),
                            clip_on=False,
                        )
                        # ax.set_label(str(diff))
                        patches.append(patch)
                        # z = patch
                    ax.legend(handles=patches)
        plt.savefig(f"inlp_beir_catplot_all_metrics.pdf")
