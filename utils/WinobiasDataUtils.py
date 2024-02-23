# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import string
from pathlib import Path

import torch
import re
import numpy as np
from sklearn.utils import shuffle
from tqdm import tqdm

from transformers import RobertaModel

pronouns = ["he", "she", "her", "him", "his", "herself", "himself"]


def clean_winobias(data):
    """
    replace pronouns in the sentence with <mask>
    """
    sentences_masked = []
    sentences = []
    for ex in data:
        clean_ex = re.split("^[0-9]+ ", ex)[1]
        clean_ex = clean_ex.strip().replace('[', '').replace(']', '')
        sentences.append(clean_ex)
        for p in pronouns:
            if f" {p} " in clean_ex or f" {p}." in clean_ex:
                clean_ex = clean_ex.replace(f" {p} ", " <mask> ").replace(f" {p}.", " <mask>.")
        sentences_masked.append(clean_ex)
    return sentences, sentences_masked


def tokenize_and_map_words(sentence, tokenizer):
    #TODO pick this up here -- it might not be necessary for contriever and only is for roberta

    # start index because the number of special tokens is fixed for each model (but be aware of single sentence input and pairwise sentence input)
    idx = 1

    splitted_sentence = sentence.split()
    # Get word by word tokenization. There is a separation between first word and other because roBERTa tokenize
    # treats differently words inside the sentence (with space before) and words in the beginning of the sentence
    # (without space before) - there is different encoding.
    enc = [tokenizer.encode(x, add_special_tokens=False, add_prefix_space=True) for x in splitted_sentence[1:]]
    enc.insert(0, tokenizer.encode(splitted_sentence[0], add_special_tokens=False, add_prefix_space=False))

    desired_output = []

    for token in enc:
        tokenoutput = []
        for ids in token:
            tokenoutput.append(idx)
            idx += 1
        desired_output.append(tokenoutput)

    return tokenizer.encode(sentence, return_tensors='pt'), desired_output


def get_wanted_indices(sentence, tokenizer, occupation):
    tokenized_sentence, words_mapping = tokenize_and_map_words(sentence, tokenizer)
    if len(occupation.split()) > 1:
        wanted_indices = []
        for word in occupation.split():
            wanted_indices += words_mapping[
                sentence.translate(str.maketrans('', '', string.punctuation)).split().index(word)]
    else:
        wanted_indices = words_mapping[
            sentence.translate(str.maketrans('', '', string.punctuation)).split().index(occupation)]

    return wanted_indices

def sort_dataset_for_probing(vectors, genders, professions):
    new_X = []
    new_z = []
    new_professions = []
    all_professions = np.unique(professions)
    np.random.shuffle(all_professions)
    for prof in all_professions:
        new_Xs = vectors[professions == prof]
        new_professions += [prof] * len(new_Xs)
        for vec in new_Xs:
            new_X.append(vec)
        for vec in genders[professions == prof]:
            new_z.append(vec)

    new_X = np.stack(new_X)
    new_z = np.stack(new_z)
    new_professions = np.stack(new_professions)

    return new_X, new_z, new_professions

def Winobias_extract_vectors_data(model, tokenizer, device, source_folder='../data/winobias',
                                  destination_folder="../data/winobias"):
    # it doesn't matter if we take the pro or anti stereotypic data because we mask the pronouns (this was verified)
    with open(source_folder + f'/pro_stereotyped_type1.txt.test') as f:
        content_t1_test = f.readlines()
    with open(source_folder + f'/pro_stereotyped_type2.txt.test') as f:
        content_t2_test = f.readlines()
    with open(source_folder + f'/pro_stereotyped_type1.txt.dev') as f:
        content_t1_dev = f.readlines()
    with open(source_folder + f'/pro_stereotyped_type2.txt.dev') as f:
        content_t2_dev = f.readlines()

    _, sentences_masked_t1_test = clean_winobias(content_t1_test)
    _, sentences_masked_t1_dev = clean_winobias(content_t1_dev)
    _, sentences_masked_t2_test = clean_winobias(content_t2_test)
    _, sentences_masked_t2_dev = clean_winobias(content_t2_dev)
    sentences_masked = sentences_masked_t1_test + sentences_masked_t1_dev + sentences_masked_t2_test + sentences_masked_t2_dev

    with open(source_folder + "/male_occupations.txt") as f:
        male_occ = f.readlines()
        male_occ = [x.strip() for x in male_occ]
    with open(source_folder + "/female_occupations.txt") as f:
        female_occ = f.readlines()
        female_occ = [x.strip() for x in female_occ]

    vectors = []
    genders = []
    professions = []
    indices = []

    with torch.no_grad():
        model.eval()
        for i, sentence in enumerate(tqdm(sentences_masked)):
            # get the occupations in the sentence
            female_occupation = list(filter(lambda x: x in sentence, female_occ))
            female_occupation = max(female_occupation, key=len)
            male_occupation = list(filter(lambda x: x in sentence, male_occ))
            male_occupation = max(male_occupation, key=len)

            if type(model) == RobertaModel:
                male_wanted_indices = get_wanted_indices(sentence, tokenizer, male_occupation)
                female_wanted_indices = get_wanted_indices(sentence, tokenizer, female_occupation)
                tokenized_sentence, words_mapping = tokenize_and_map_words(sentence, tokenizer)

                tokenized_sentence = tokenized_sentence.to(device)
                male_embed = model(tokenized_sentence)[0][0][male_wanted_indices].cpu().detach().numpy().mean(axis=0)
                female_embed = model(tokenized_sentence)[0][0][female_wanted_indices].cpu().detach().numpy().mean(axis=0)
            else: # contriever model
                #TODO, abandoned
            vectors.append(male_embed)
            vectors.append(female_embed)

            # print(words_mapping[male_wanted_indices])
            # print(tokenized_sentence)
            # print(tokenizer.decode(tokenized_sentence[0][male_wanted_indices]))
            # print(tokenizer.decode(tokenized_sentence[0][female_wanted_indices]))

            genders.append("M")
            professions.append(male_occupation)
            genders.append("F")
            professions.append(female_occupation)
            indices.append(i)
            indices.append(i)

        vectors = np.array(vectors)
        genders = np.array(genders)
        professions = np.array(professions)

    vectors, genders, professions = sort_dataset_for_probing(vectors, genders, professions)
    Path(destination_folder).mkdir(parents=True, exist_ok=True)
    torch.save({"X": vectors, "z": genders, "original index": indices, "professions": professions},
               destination_folder + f"/vectors_roberta-base.pt")
