# Overview

This repo accompanies the paper _MULTICONTRIEVERS: ANALYSIS OF DENSE RETRIEVAL REPRESENTATIONS_ [(link here)](https://openreview.net/forum?id=JWHf7lg8zM&noteId=tEU5I2TzCc).

The goal of this project is to try to determine what kind of information is in dense representations, and how it affects retriever quality and bias.

Below are notes on how to run the experiments.
Many of them are config driven and lists of models and datasets can be found in `config/`

## Licenses

Unless noted below, all code in this repository is licensed under the Creative Commons Non-Commercial License, listed in the LICENSE file.

The paper builds upon work from five other projects. Two of the codebases are sufficently modified that they are included in here, the other will need to be downloaded from source and incorporated to reproduce this work.

Included here is significantly modified code taken from:

- [Minimum Description Length Probing](https://github.com/lena-voita/description-length-probing), by Lena Voita. This code has no license.
- [How Gender Debiasing Affects Internal Model Representations, and Why It Matters](https://github.com/orgadhadas/gender_internal), by Hadas Orgad, contributed to and co-authored by me (Seraphina Goldfarb-Tarrant). This code has an MIT license and in this repo is under `src`

Not included in this release are three large repositories that have entirely modular modifications. To reproduce this work, install them from source and follow the instructions given below to incorporate them into this repo:

- [Contriever](https://github.com/facebookresearch/contriever), by Gautier Izacard. This code is owned by facebook research under a Creative Commons non-commercial license. We used this to train Contriever models.
- [BEIR](https://github.com/beir-cellar/beir), by Nandan Thakur at the University of Waterloo. This code is under an Apache 2.0 license. We use this to evaluate retriever models, it is the industry standard way to do so.
- [Iterative Nullspace Projection](https://github.com/shauli-ravfogel/nullspace_projection) by Shauli Ravfogel, at Bar-Ilan University. This code has no license. We use it for experiments in removing gender information directly from representations.

## Data

- Checkpoints: https://dl.fbaipublicfiles.com/multicontriever-analysis/checkpoints
- Analysis Data: https://dl.fbaipublicfiles.com/multicontriever-analysis/analysis

## Use the multi-contrievers

There are 25 contrievers train on top of 25 multiberts, which were released in the [multiberts paper](https://arxiv.org/abs/2106.16163).

All contriever checkpoints by seed can be found at `config/contrievers.yaml` (a dict of format seed: absolute path).
e.g. to get seed 10 path, do:

```
SEED=10
model_config = yaml.load(open(args.model_config), Loader=yaml.FullLoader)
my_path = model_config["model_paths"][SEED]
```

The path you will get is the final best model, a checkpoint was saved every 50k steps. If you want to look at training dynamics, change the format of the path to end in `step-NUMSTEPS` instead of `latest`.

### Getting contriever representations

My env is at `/home/seraphina/.conda/envs/sgt/bin/python`

clone my contriever [codebase](https://github.com/seraphinatarrant/contriever) and cd into it.
If you already have the standard contriever release from facebook research [here](https://github.com/facebookresearch/contriever) it isn't necessary to grab mine, I made no changes to basic inference code.

```
from src.contriever import load_retriever

model, tokenizer, _ = load_retriever(model_path)
tokenizer.model_max_length = 512  # the multiberts don't have this in their config so sequences can end up too long

# tokenize & embed
encoded_dict = tokenizer(inputs, padding=True, truncation=True, return_tensors="pt", return_token_type_ids=False)
vectors = model(encoded_dict["input_ids"], attention_mask=encoded_dict["attention_mask"])
```

To get representations from the same bert that the contriever was trained from:

```
from transformers import AutoTokenizer, BertModel
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained(f"google/multiberts-seed_{SEED}")
```

And then tokenize and embedd with the same code above.

## Run probing

All steps for probing involve:

1. Processing raw text into tokenised spans and labels. Often this only has to happen once since many models share the same tokenizer.
2. Transforming the output of step 1 into representations and labels.
3. Running a probe on the output of step 2.

All of these steps can be seen in `run_probe_biasinbios.sh`, but in practice I tend to run Step 1 once for a batch and then launch a job that runs steps 2 & 3.

Bash script for extracting contriever representations in bulk:

1. extract_tokens_new_contriever.sh (run once as tokenizer is shared)
2. extract_reps_and run_new_contriever.sh SEED for each seed
3. Check wandb summary for the results!

## Evaluate contrievers on BEIR

`run_beir_eval.py` launches a beir evaluation for all models specified in `config/contrievers.yaml` and datasets specified in `config/beir_datasets.yaml`,
(which can be overridden to other config paths with script args).

`run_beir_eval.slurm SEED MODEL_PATH` runs the beir evaluation for one model sans config.

## Graph results & get statistics

BEIR results vs. compression

### Process raw results

First process into a dataframe.
`python process_beir_logs.py --log_files LOG1 LOG2 LOG3...etc --output_dir A_GOOD_PLACE.pkl`

This will read compression information from wandb and beir results from the logs and create one dataframe that can be used to visualise results.

### graph results

`python graph_results.py` will save pdf scatterplots of compression vs. metric. Set the `--model` and `metric` args to look at different combos.
The `by_dataset` flag breaks out the plots per each dataset (rather than an average, which is the default) and saves one graph per each.

### get correlation and stat sig

`python run_stats.py` will run spearman and pearson correlation with the flag `--stat correlation` or run a permutation test of stat sig with the flag `--stat significance`. Similarly to graphing, set the `--model` and `metric` args.
