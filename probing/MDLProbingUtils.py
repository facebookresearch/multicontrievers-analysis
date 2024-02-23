import argparse
import os
from dataclasses import dataclass
from os.path import join
from typing import List
from collections import Counter

import torch
from torch import nn
from transformers import set_seed
import wandb
from transformers.integrations import is_wandb_available

from probing.mdl import OnlineCodeMDLProbe


@dataclass
class OnlineCodingExperimentResults:
    name: str
    uniform_cdl: float
    online_cdl: float
    compression: float
    report: dict
    fractions: List[float]

def general_probing_args(parser):
    parser.add_argument('--seed', type=int, help='the random seed to check on', required=True)
    parser.add_argument('--model_seed', type=int, help='the random seed used to train the model', required=True)
    parser.add_argument('--embedding_size', type=int, help='embedding size', default=768)
    parser.add_argument('--batch_size', type=int, help='batch size to train the probe', default=16)
    parser.add_argument('--probe_type', type=str, help='linear probe or MLP', choices=['linear', 'mlp'], default='linear')

def general_MDL_args():
    parser = argparse.ArgumentParser(description='Probe vectors for gender of example using MDL probes.')
    general_probing_args(parser)
    parser.add_argument('--mdl_fractions', nargs='+', type=int, help='linear probe of MLP',
                        default=[2.0, 3.0, 4.4, 6.5, 9.5, 14.0, 21.0, 31.0, 45.7, 67.6, 100])

    return parser


def build_probe(input_size, num_classes, probe_type='mlp'):
    probes = {
        'mlp': lambda: nn.Sequential(
            nn.Linear(input_size, input_size // 2),
            nn.Tanh(),
            nn.Linear(input_size // 2, num_classes)
        ),
        'linear': lambda: nn.Linear(input_size, num_classes)
    }
    return probes[probe_type]()


def create_probe(args, n_labels):
    return build_probe(args.embedding_size, n_labels, probe_type=args.probe_type)


def run_MDL_probing(args, load_fn, task_name, shuffle):
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset, val_dataset, test_dataset, n_labels = load_fn(args)
    # print training and test distributions
    train_counter, test_counter = Counter([i.item() for i in train_dataset.tensors[1]]), \
                                  Counter([i.item() for i in test_dataset.tensors[1]])
    n_samples_train, n_samples_test = train_dataset.tensors[1].shape[0], test_dataset.tensors[1].shape[0]
    
    train_stats = "\n".join([f"{key}: {val/n_samples_train*100}%" for key, val in train_counter.items()])
    test_stats = "\n".join([f"{key}: {val/n_samples_test*100}%" for key, val in test_counter.items()])
    print(f"Train size: {train_dataset.tensors[1].shape[0]} Test size: {test_dataset.tensors[1].shape[0]}")
    print(f"Train stats:\n{train_stats}\nTest stats:\n{test_stats}\n")
    if is_wandb_available():
        wandb.config.update({
            "n_labels": n_labels,
            "n_train": n_samples_train,
            "n_test": n_samples_test,
            "train_label_stats": train_stats,
            "test_label_stats": test_stats
            })

    online_code_probe = OnlineCodeMDLProbe(lambda: create_probe(args, n_labels), args.mdl_fractions, device=device)

    reporting_dir = join(os.getcwd(), 'mdl_results/')
    if not os.path.exists(reporting_dir):
        os.makedirs(reporting_dir)
        
    reporting_root = join(reporting_dir, f'online_coding_{task_name}.pkl')
    uniform_cdl, online_cdl = online_code_probe.evaluate(train_dataset, test_dataset, val_dataset,
                                                         reporting_root=reporting_root, verbose=True, device=device,
                                                         train_batch_size=args.batch_size, shuffle=shuffle)
    compression = round(uniform_cdl / online_cdl, 2)
    report = online_code_probe.load_report(reporting_root)

    exp_results = OnlineCodingExperimentResults(
        name=task_name,
        uniform_cdl=uniform_cdl,
        online_cdl=online_cdl,
        compression=compression,
        report=report,
        fractions=args.mdl_fractions
    )

    return exp_results