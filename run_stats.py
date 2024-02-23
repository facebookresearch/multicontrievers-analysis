# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Example command:
python run_stats.py --stat correlation --metric Recall@100 --data_path /home/seraphina/contriever/beir/results/contriever/all_seeds_biasinbios_profession.pkl
"""

import argparse
import os
import pickle
from re import M
from tqdm import tqdm
from collections import defaultdict

import pandas as pd
from scipy.stats import pearsonr, spearmanr, permutation_test, wilcoxon
from sklearn.metrics import r2_score
import numpy as np
import ipdb


def setup_argparse():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data_path",
        help="file to load: Dataframe (correlation) or dict (significance)",
        default="/home/seraphina/contriever/beir/results/contriever/all_metrics_all_seeds.pkl",
    )
    p.add_argument(
        "--stat", default="correlation", choices=["correlation", "significance"]
    )
    p.add_argument("--model", type=str, default="contriever")
    p.add_argument(
        "--model2", type=str, default="bert", help="used to compare models for stat sig"
    )
    p.add_argument("--metric", type=str, default="NDCG@10")
    p.add_argument(
        "--data_type",
        type=str,
        default="raw",
        choices=["raw", "scrubbed"],
        help="type of data compression tested on",
    )
    p.add_argument("--by_dataset", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = setup_argparse()

    if args.stat == "correlation":
        df = pd.read_pickle(args.data_path)
        df = df[df["model"] == args.model]
        df = df[df["data_type"] == args.data_type]
        metric_df = df[df["metric"] == args.metric]

        # print out correlation average, and then by dataset
        if not args.by_dataset:
            avg_df = metric_df[metric_df["dataset"] == "all"]
            perfs, comps = avg_df["value"].values, avg_df["compression"].values
            print("Correlations for macro average BEIR over all datasets:")
            print(pearsonr(comps, perfs))
            print(spearmanr(comps, perfs))

        print("Correlations for for each BEIR dataset:")
        all_ds_names = metric_df["dataset"].unique()
        for ds_n in tqdm(all_ds_names):
            # if ds_n == "all":
            #    continue
            # print(ds_n)
            this_ds = metric_df[metric_df["dataset"] == ds_n]
            perfs, comps = this_ds["value"].values, this_ds["compression"].values
            pearson = pearsonr(comps, perfs)
            spearman = spearmanr(comps, perfs)
            r2 = r2_score(comps, perfs)
            print("{} {} {} Pearson".format(ds_n, pearson.statistic, pearson.pvalue))
            print(
                "{} {} {} Spearman".format(ds_n, spearman.correlation, spearman.pvalue)
            )
            # print("{} {} R2".format(ds_n, r2)) TODO make this work

    elif args.stat == "significance":
        # data path defaults to {dataset}_type2model2seed2compression.pkl
        with open(args.data_path, "rb") as fin:
            t2m2s2c = pickle.load(fin)

        m2s2c = t2m2s2c[args.data_type]
        seed2comp = m2s2c[args.model]
        compare_seed2comp = m2s2c[args.model2]

        # TODO decide what to compare?
        # x = list(seed2comp.values())
        x, y = [], []
        s = []
        all_seeds = sorted(seed2comp.keys(), key=lambda x: int(x.split("_")[1]))
        for seed in all_seeds:
            if seed not in compare_seed2comp:
                continue
            comp = seed2comp[seed]
            x.append(comp)
            y.append(compare_seed2comp[seed])
            s.append(seed)
        print(s)
        print(x)
        print(y)

        # def statistic(x, y):
        #     return np.mean(x) == np.mean(y)

        # res = permutation_test((x, y), statistic, vectorized=False,
        #                permutation_type='independent')
        # r, pvalue, null = res.statistic, res.pvalue, res.null_distribution

        res = wilcoxon(x, y)
        print(f"Statistical significance for {args.model} and {args.model2}")
        print(res)
