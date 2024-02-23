import argparse
import wandb
import sys
from random import shuffle

from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier
from transformers import set_seed
import numpy as np

from probing.MDLProbingUtils import build_probe, OnlineCodingExperimentResults, create_probe, run_MDL_probing, general_MDL_args, \
    general_probing_args

sys.path.append('../src')
from utils.ScriptUtils import load_winobias_vectors


def parse_args():
    parser = argparse.ArgumentParser(description='Probe vectors for gender of example, output accuracy.')
    general_probing_args(parser)
    parser.add_argument('--model', required=True, type=str, help='the model type',
                        choices=["basic", "finetuned", "random"])
    parser.add_argument('--training_task', type=str, help='the model type',
                        choices=["biasinbios", "coref"], default=None)
    parser.add_argument('--training_balanced', type=str, help='balancing of the training data',
                        choices=["balanced", "imbalanced", "subsampled", "oversampled", "original"], default=None)
    parser.add_argument('--training_data', default=None, type=str, help="bias in bios training data type",
                        choices=["raw", "scrubbed"])
    parser.add_argument('--model_number', '-n', type=int, help='the model number to check on', required=False, default=None)
    parser.add_argument('--embedding_seed', type=int, default=None, required=False, help='the random seed the model was trained on')

    args = parser.parse_args()
    print(args)
    return args


def init_wandb(args):
    wandb.init(project="Winobias probing", config={
        "architecture": "RoBERTa",
        "seed": args.seed,
        "embedding_seed": args.embedding_seed,
        "training_task": args.training_task,
        "training balancing": args.training_balanced,
        "training_data": args.training_data,
        "model": args.model,
        "probe_type": args.probe_type,
        "batch_size": args.batch_size,
        "model_number": args.model_number,
    })

def split_by_professions(data, train_frac = 0.8):
    unique_prof = np.unique(data['professions'])
    shuffle(unique_prof)

    X_train = []
    y_train = []
    X_test = []
    y_test = []
    i = 0
    while i < len(unique_prof) * train_frac:
        prof = unique_prof[i]
        X_train.append(data['X'][data['professions'] == prof])
        y_train.append(data['z'][data['professions'] == prof])
        i += 1
    while i < len(unique_prof):
        prof = unique_prof[i]
        X_test.append(data['X'][data['professions'] == prof])
        y_test.append(data['z'][data['professions'] == prof])
        i += 1

    X_train = np.concatenate(X_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)
    X_test = np.concatenate(X_test, axis=0)
    y_test = np.concatenate(y_test, axis=0)

    return X_train, y_train, X_test, y_test

def main():
    args = parse_args()
    set_seed(args.seed)
    init_wandb(args)

    data, _, _ = load_winobias_vectors(args, preprocess=False)
    data = {
        "X": data['X'],
        "z": data['z'],
        "professions": data['professions'],
    }

    X_train, y_train, X_test, y_test = split_by_professions(data, train_frac=0.8)

    if args.probe_type == 'linear':
        hidden_layer_sizes = ()
    else:
        hidden_layer_sizes = (args.embedding_size // 2, args.embedding_size // 2)

    clf = MLPClassifier(random_state=args.seed, batch_size=args.batch_size, hidden_layer_sizes=hidden_layer_sizes,
                        activation='tanh', max_iter=2000).fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    wandb.summary['acc'] = score

    maj = 'F' if len(y_train[y_train == 'F']) > len(y_train[y_train == 'M']) else 'M'
    majority_acc = len(y_test[y_test == maj]) / len(y_test)
    wandb.summary['majority_acc_test'] = majority_acc

    y_pred = clf.predict(X_test)
    wandb.summary['f1'] = f1_score(y_test, y_pred, average='macro')

if __name__ == '__main__':
    main()
