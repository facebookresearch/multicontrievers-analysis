# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import random
import os

import torch
from transformers import RobertaModel, RobertaForMaskedLM, RobertaConfig, AutoModel
from tr
ansformers import RobertaTokenizer, AutoTokenizer
import numpy as np

from contriever.src.contriever import Contriever

from utils.WinobiasDataUtils import Winobias_extract_vectors_data

N_LABELS = 28

parser = argparse.ArgumentParser(description='Extracting vectors from a model representation.')
parser.add_argument('--model', required=True, type=str, help='the model type',
                    choices=["basic", "finetuned", "random", "contriever"])
parser.add_argument('--training_balanced', type=str, help='type of training data used to train the model',
                    choices=["balanced", "imbalanced", "anon", "CA"], default=None)
parser.add_argument('--number', '-n', default=0, type=int, help='the number of model')
parser.add_argument('--seed', '-s', default=0, type=int, help='seed for random roberta')
parser.add_argument('--contriever_type', '-ct', type=str, choices=["contriever", "contriever-msmarco", "mcontriever", "contriever-msmarco"], default="contriever")

args = parser.parse_args()
print("model:", args.model)
print("trained on:", args.training_balanced)
print("number:", args.number)
print("seed:", args.seed)

model_version = 'roberta-base'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

source_folder = "data/winobias"
dest_folder_base = "data/winobias/extracted_vectors/"

seed = args.seed
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
np.random.seed(seed)

if args.model == "contriever":
    dest_folder_base = os.path.join(dest_folder_base, "contriever")
    tokenizer = AutoTokenizer.from_pretrained("facebook/contriever")
    model = Contriever.from_pretrained(f"facebook/{args.contriever_type}")
    destination_folder = os.path.join(dest_folder_base, args.contriever_type)

    
else:
    tokenizer = RobertaTokenizer.from_pretrained(model_version)
    dest_folder_base = os.path.join(dest_folder_base, "roberta-base")
    
    if args.model == "basic":
        model = RobertaModel.from_pretrained(model_version, output_attentions=True, output_hidden_states=True)
        destination_folder = os.path.join(dest_folder_base, "basic")
    if args.model == "random":
        configuration = RobertaConfig()
        model = AutoModel.from_config(configuration)
        model.resize_token_embeddings(len(tokenizer))
        destination_folder = f"data/winobias/extracted_vectors/roberta-base/random/seed_{seed}"
    if args.model == "finetuned":
        # config_name = "test_roberta_base" if args.training_balanced == "imbalanced" else "test_roberta_base_balanced_model"
        # saved_suffix = args.training_balanced + str(args.number)
        # runner = Runner(config_name, 0)
        # model = runner.initialize_model(saved_suffix).bert
        model = RobertaModel.from_pretrained(model_version, output_attentions=True, output_hidden_states=True)
        model.load_state_dict(torch.load(f"embedding_models/roberta-base/{args.training_balanced}/{args.number}/model.pt"))
        model_version = 'roberta-base'
        destination_folder = f"data/winobias/extracted_vectors/roberta-base/finetuned/{args.training_balanced}/number_{args.number}"

model = model.to(device)
Winobias_extract_vectors_data(model, tokenizer, device, source_folder, destination_folder)
