import argparse
import random
import time
import os
import re
from tqdm import tqdm

import numpy as np
from sklearn.linear_model import (
    SGDClassifier,
    SGDRegressor,
    Perceptron,
    LogisticRegression,
)
from sklearn.svm import LinearSVC, SVC
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import torch
from datasets import ClassLabel

from src import debias


def get_projection_matrix(
    num_clfs,
    X_train,
    Y_train_gender,
    X_dev,
    Y_dev_gender,
    Y_train_task,
    Y_dev_task,
    dim,
):
    is_autoregressive = True
    min_acc = 0.0
    # noise = False
    dim = 768
    n = num_clfs
    # random_subset = 1.0
    start = time.time()
    TYPE = "svm"
    MLP = False  # TODO why is this here

    if MLP:
        x_train_gender = np.matmul(x_train, clf.coefs_[0]) + clf.intercepts_[0]
        x_dev_gender = np.matmul(x_dev, clf.coefs_[0]) + clf.intercepts_[0]
    else:
        x_train_gender = x_train.copy()
        x_dev_gender = x_dev.copy()

    if TYPE == "sgd":
        gender_clf = SGDClassifier
        params = {
            "loss": "hinge",
            "penalty": "l2",
            "fit_intercept": False,
            "class_weight": None,
            "n_jobs": 32,
        }
    else:
        gender_clf = LinearSVC
        params = {
            "penalty": "l2",
            "C": 0.01,
            "fit_intercept": True,
            "class_weight": None,
            "dual": False,
        }

    P, rowspace_projections, Ws = debias.get_debiasing_projection(
        gender_clf,
        params,
        n,
        dim,
        is_autoregressive,
        min_acc,
        X_train,
        Y_train_gender,
        X_dev,
        Y_dev_gender,
        Y_train_main=Y_train_task,
        Y_dev_main=Y_dev_task,
        by_class=True,
    )
    print("time: {}".format(time.time() - start))
    return P, rowspace_projections, Ws


def train_classifier(x_train, y_train, x_test=None, y_test=None):
    random.seed(0)
    np.random.seed(0)

    clf = LogisticRegression(
        warm_start=True,
        penalty="l2",
        solver="saga",
        multi_class="multinomial",
        fit_intercept=False,
        verbose=5,
        n_jobs=90,
        random_state=1,
        max_iter=7,
    )

    start = time.time()
    idx = np.random.rand(x_train.shape[0]) < 1.0
    clf.fit(x_train[idx], y_train[idx])
    print("time: {}".format(time.time() - start))
    print(f"Train acc: {clf.score(x_train, y_train)}")
    if x_test is not None and y_test is not None:
        print(f"Test acc: {clf.score(x_test, y_test)}")
    return clf


def tsne_by_gender(vecs, labels, title, words=None, save_dir=None):
    tsne = TSNE(n_components=2, random_state=0)
    vecs_2d = tsne.fit_transform(vecs)
    num_labels = len(set(labels.tolist()))

    names = list(set(labels))  # ["class {}".format(i) for i in range(num_labels)]
    plt.figure(figsize=(6, 5))
    colors = "b", "r", "orange"
    markers = ["o", "s"]

    for i, c, label, marker in zip(set(labels.tolist()), colors, names, markers):
        print(len(vecs_2d[labels == i, 0]))
        plt.scatter(
            vecs_2d[labels == i, 0],
            vecs_2d[labels == i, 1],
            c=c,
            label="Female" if label == 1 else "Male",
            alpha=0.45,
            marker=marker,
        )
    plt.legend(fontsize=15, loc="upper right")
    plt.title(title, fontsize=15)

    if words is not None:
        k = 60
        for i in range(k):
            j = np.random.choice(range(len(words)))
            label = labels[i]
            w = words[j]
            x, y = vecs_2d[i]
            plt.annotate(w, (x, y), size=10, color="black" if label == 1 else "black")
    title += ".pdf"
    save_file = os.path.join(save_dir, title) if save_dir else title
    plt.savefig(save_file)
    # plt.show()
    return vecs_2d


def setup_argparse():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", help="directory with vectors to run on and save to")
    p.add_argument("--model_type", choices=["contriever_new", "bert"])
    p.add_argument("--display", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = setup_argparse()
    print(args)

    vec_path = os.path.join(args.data_dir, "{}.vectors_raw_" + f"{args.model_type}.pt")

    train = torch.load(vec_path.format("train"))
    dev = torch.load(vec_path.format("dev"))
    test = torch.load(vec_path.format("test"))

    gender_labels = ClassLabel(names=["m", "f"])  # to ensure always m:0 f:1
    all_prof = set(train.get("y")) | set(dev.get("y")) | set(test.get("y"))
    prof_labels = ClassLabel(names=all_prof)

    # get X and Y
    device = "cpu"
    x_train = train["X"].to(device).numpy()
    x_dev = dev["X"].to(device).numpy()
    x_test = test["X"].to(device).numpy()

    y_train = np.array([prof_labels.str2int(entry) for entry in train["y"]])
    y_dev = np.array([prof_labels.str2int(entry) for entry in dev["y"]])
    y_test = np.array([prof_labels.str2int(entry) for entry in test["y"]])

    # profession classifier
    orig_classifier = train_classifier(x_train, y_train, x_test, y_test)

    # INLP
    num_clfs = 300
    y_dev_gender = np.array([gender_labels.str2int(d) for d in dev["z"]])
    y_train_gender = np.array([gender_labels.str2int(d) for d in train["z"]])
    y_test_gender = np.array([gender_labels.str2int(d) for d in test["z"]])

    idx = np.random.rand(x_train.shape[0]) < 1.0
    P, rowspace_projections, Ws = get_projection_matrix(
        num_clfs,
        x_train[idx],
        y_train_gender[idx],
        x_dev,
        y_dev_gender,
        y_train,
        y_dev,
        300,
    )

    # save projection files and then also modify existing files and save also
    print("Saving matrix, rowspace projections, and classifiers...")
    torch.save(P, os.path.join(args.data_dir, "projection_matrix.pt"))
    torch.save(
        rowspace_projections, os.path.join(args.data_dir, "rowspace_projections.pt")
    )
    torch.save(Ws, os.path.join(args.data_dir, "debiasing_classifiers.pt"))
    print("Saving inlp representations...")
    for data, split in tqdm(zip([dev, test, train], ["dev", "test", "train"])):
        input_x = data["X"].cpu().numpy()  # need numpy broadcasting
        projection = (P.dot(input_x.T)).T
        data["X"] = torch.from_numpy(projection).float().to("cuda")
        new_filename = re.sub("raw", "inlp", vec_path.format(split))
        torch.save(data, new_filename)

    # optionally rerun a classifier
    clf = LogisticRegression(
        warm_start=True,
        penalty="l2",
        solver="sag",
        multi_class="multinomial",
        fit_intercept=True,
        verbose=10,
        max_iter=3,
        n_jobs=64,
        random_state=1,
    )
    # clf = SGDClassifier()
    # P_rowspace = np.eye(768) - P
    # mean_gender_vec = np.mean(P_rowspace.dot(x_train.T).T, axis = 0)

    print("Accuracy on gender classification for projected data")
    print(clf.fit((P.dot(x_train.T)).T, y_train))

    # display tsne
    if args.display:
        n = 2000
        for prof in [
            "nurse",
            "professor",
            "physician",
            "accountant",
            "dj",
            "dietitian",
        ]:
            idx = np.random.rand(x_test.shape[0]) < 0.1
            prof_idx = y_dev == prof_labels.str2int(prof)
            prof_upper = prof[0].upper() + prof[1:]
            tsne_by_gender(
                x_dev[prof_idx][:n],
                y_dev_gender[prof_idx][:n],
                "{} (Original)".format(prof_upper),
                save_dir=args.data_dir,
            )
            tsne_by_gender(
                (x_dev[prof_idx].dot(P))[:n],
                y_dev_gender[prof_idx][:n],
                "{} (Projected)".format(prof_upper),
                save_dir=args.data_dir,
            )
        tsne_by_gender(
            x_dev[:n],
            y_dev_gender[:n],
            "All (Original)".format(prof_upper),
            save_dir=args.data_dir,
        )
        tsne_by_gender(
            (x_dev.dot(P))[:n],
            y_dev_gender[:n],
            "All (Projected)".format(prof_upper),
            save_dir=args.data_dir,
        )
