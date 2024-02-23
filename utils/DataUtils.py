# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pickle
import re
import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import torch
from torch.utils.data import ConcatDataset, TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import RobertaTokenizer, RobertaModel, BertModel, DistilBertModel


dataset2filebase = {
    "biasinbios": "data/biasbios/vectors_extracted_from_trained_models/",
    "wizard": "data/md_gender/wizard/vectors_extracted_from_trained_models/",
    "wizard_binary": "data/md_gender/wizard_binary/vectors_extracted_from_trained_models",
    "wikipedia": "data/md_gender/wikipedia/vectors_extracted_from_trained_models",
    "wikipedia_binary": "data/md_gender/wikipedia_binary/vectors_extracted_from_trained_models",
}


def process_biasinbios(ds: dict, datatype: str) -> tuple:
    inputs, labels, genders = [], [], []
    for r in tqdm(ds):
        if datatype == "name":  # TODO doesn't currently support name
            sent = " ".join(r["name"])
        elif datatype == "scrubbed":
            sent = r[
                "text_without_gender"
            ]  # no start needed since scrubbed already removes first line
        else:
            sent = r["text"][r["start"] :]

        inputs.append(sent)
        labels.append(r["p"])  # profession
        genders.append(r["g"])
    return inputs, labels, genders


def process_wizard(ds: dict, datatype: str) -> tuple:
    inputs, labels, genders = [], [], []
    for r in tqdm(ds):
        if datatype == "scrubbed":
            sys.exit("Doesn't currently support scrubbed")
        else:
            sent = r["text"]

        inputs.append(sent)
        labels.append(r["chosen_topic"])
        genders.append(r["gender"])
    return inputs, labels, genders


def process_wikipedia(ds: dict, datatype: str) -> tuple:
    # does not have labels so labels will be empty.
    inputs, labels, genders = [], [], []
    for r in tqdm(ds):
        if datatype == "scrubbed":
            sys.exit("Doesn't currently support scrubbed")
        else:
            sent = r["text"]

        inputs.append(sent)
        genders.append(r["gender"])
    return inputs, labels, genders


def process_dataset(dataset_name, ds, datatype):
    if dataset_name == "biasinbios":
        inputs, labels, genders = process_biasinbios(ds, datatype)
    elif "wizard" in dataset_name:  # captures binary and non
        inputs, labels, genders = process_wizard(ds, datatype)
    elif "wikipedia" in dataset_name:
        inputs, labels, genders = process_wikipedia(ds, datatype)

    return inputs, labels, genders


def divide_chunks(items: tuple, n: int = 300000):
    x, y, z = items  # inputs, labels, genders
    for i in range(0, len(x), n):
        yield x[i : i + n], y[i : i + n], z[i : i + n]


def batch_encode(generator, tokenizer):
    for text, label, gender in generator:
        yield (
            tokenizer(
                text,
                padding=True,
                truncation=True,
                return_tensors="pt",
                return_token_type_ids=False,
            ),
            label,
            gender,
        )


def extract_tokens(
    datapath: str,
    datatype: str,
    tokenizer,
    model_type: str,
    batch: bool = False,
    dataset: str = "biasinbios",
    truncate=False,
    tensor_dataset: bool = False,
):
    with open(datapath, "rb") as f:
        ds = pickle.load(f)

    outfile = (
        f"{os.path.splitext(datapath)[0]}.tokens_{datatype}_{model_type}.pt"
        if batch
        else f"{os.path.splitext(datapath)[0]}.tokens_{datatype}_{model_type}_unbatched.pt"
    )

    inputs, labels, genders = process_dataset(dataset, ds, datatype)

    max_batch = 300000
    if len(inputs) < max_batch and not tensor_dataset:
        if model_type == "roberta":
            if not truncate:
                encoded_dict = tokenizer(
                    inputs,
                    add_special_tokens=True,
                    padding=True,
                    return_attention_mask=True,
                    return_token_type_ids=False,
                )
            else:
                max_length = 128
                print("truncating to max len")
                encoded_dict = tokenizer(
                    inputs,
                    add_special_tokens=True,
                    padding="max_length",
                    return_token_type_ids=False,
                    max_length=max_length,
                    truncation=True,
                    return_attention_mask=True,
                )
        else:
            encoded_dict = tokenizer(
                inputs,
                padding=True,
                truncation=True,
                return_tensors="pt",
                return_token_type_ids=False,
            )

        encoded_dict["input_ids"] = encoded_dict["input_ids"].int()
        encoded_dict["attention_mask"] = encoded_dict["attention_mask"].char()
        # Convert the lists into numpy arrays.
        attention_masks = encoded_dict["attention_mask"]
        labels = np.array(labels)
        genders = np.array(genders)
        if not batch:
            input_ids = np.array(encoded_dict["input_ids"])
            outdict = {
                "X": input_ids,
                "masks": attention_masks,
                "y": labels,
                "z": genders,
            }
        else:
            outdict = {"X": encoded_dict, "y": labels, "z": genders}
        torch.save(outdict, outfile)

    else:
        #  TODO document that this currently is "extreme space saving" and thus a different format
        print(f"Chunking file as it is too long: {len(inputs)}")
        all_ds = []
        for encoded in tqdm(
            batch_encode(
                divide_chunks((inputs, labels, genders), n=max_batch), tokenizer
            )
        ):
            encoded_dict, _, genders = encoded

            x = encoded_dict["input_ids"].int()
            x_mask = encoded_dict["attention_mask"].char()
            z = torch.tensor(genders).char()
            all_ds.append(TensorDataset(x, x_mask, z))
        print("Saving Tensor Dataset")
        torch.save(ConcatDataset(all_ds), outfile)
        # torch.save({"X": encoded_dict, "y": labels, "z": genders}, f"{outfile}")


def extract_vectors(
    data_type: str,
    model,
    dest_path: str,
    data_path: str,
    batch: bool = False,
    tensor_dataset: bool = False,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    data = torch.load(data_path)

    filename = os.path.split(data_path)[1]
    filename = re.sub("tokens", "vectors", filename)
    Path(dest_path).mkdir(parents=True, exist_ok=True)
    outfile = os.path.join(dest_path, filename)

    vectors = []
    labels = []
    genders = []
    if tensor_dataset:  # already in tensor form of X, X_mask, Z
        with torch.no_grad():
            dataloader = DataLoader(data, batch_size=512)
            for b in tqdm(dataloader):
                b = tuple(t.to(device) for t in b)
                v = model(b[0], attention_mask=b[1])
                if type(model) in [BertModel, RobertaModel]:
                    v = v.last_hidden_state[:, 0, :].cpu().detach()
                vectors.append(v)
                genders.append(b[2])
            vectors = torch.cat(vectors, dim=0)
            genders = torch.cat(genders, dim=0)
            print(f"Saving to {outfile}")
            torch.save(TensorDataset(vectors, genders), outfile)
    else:
        input_data = data["X"]
        y = data["y"]
        z = data["z"]
        with torch.no_grad():
            model.eval()
            if not batch:
                X = torch.tensor(input_data).to(device)
                masks = torch.tensor(data["masks"]).to(device)
                for i, input_ids in enumerate(tqdm(X)):
                    if type(model) in [RobertaModel, BertModel, DistilBertModel]:
                        if data_type == "name":
                            v = (
                                model.embeddings(input_ids, attention_mask=masks[i])[0]
                                .mean(dim=0)
                                .cpu()
                                .detach()
                                .numpy()
                            )  # the zero indexing is to undo the unsqueeze (so not necessary in batch)
                        else:
                            v = (
                                model(
                                    input_ids.unsqueeze(0),
                                    attention_mask=masks[i].unsqueeze(0),
                                )
                                .last_hidden_state[:, 0, :][0]
                                .cpu()
                                .detach()
                                .numpy()
                            )
                    else:
                        v = (
                            model(
                                input_ids.unsqueeze(0),
                                attention_mask=masks[i].unsqueeze(0),
                            )
                            .cpu()
                            .detach()
                            .numpy()
                        )
                    vectors.append(v)
                    labels.append(y[i])
                    genders.append(z[i])
                vectors = np.array(vectors)
                labels = np.array(labels)
                genders = np.array(genders)
            else:  # process in chunks. probably works with roberta but not tested with that. Does work with Bert
                # 100-300 chunks for contriever saves about 30% of time
                num_samples = len(input_data["input_ids"])
                start_idx = 0
                batch_size = 300
                for last_idx in tqdm(range(batch_size, num_samples, batch_size)):
                    input_ids = input_data["input_ids"][start_idx:last_idx, :].to(
                        device
                    )
                    attns = input_data["attention_mask"][start_idx:last_idx, :].to(
                        device
                    )
                    v = model(input_ids, attention_mask=attns)
                    if type(model) in [BertModel, RobertaModel, DistilBertModel]:
                        v = v.last_hidden_state[:, 0, :].cpu().detach()
                    vectors.append(v)
                    start_idx = last_idx
                # catches remainder
                input_ids = input_data["input_ids"][start_idx:, :].to(device)
                attns = input_data["attention_mask"][start_idx:, :].to(device)
                v = model(input_ids, attention_mask=attns)
                if type(model) in [BertModel, RobertaModel, DistilBertModel]:
                    v = v.last_hidden_state[:, 0, :].cpu().detach()
                vectors.append(v)
                vectors = torch.cat(vectors, dim=0)
                labels = y
                genders = z
        assert (
            len(y) == len(z) and len(vectors) == len(y)
        ), f"mismatch between num samples {len(vectors)} and num labels {len(y), {len(z)}}."
        print(f"Saving to {outfile}")
        torch.save({"X": vectors, "y": labels, "z": genders}, outfile)
