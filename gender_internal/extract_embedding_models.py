import argparse
from pathlib import Path
import torch
from Models import roBERTa_classifier
from Trainer import load_checkpoint

N_LABELS = 28

parser = argparse.ArgumentParser(description='Extract the embedding model from the coref model.')
parser.add_argument('--training_data', type=str, help='type of training data used to train the model',
                    choices=["raw", "scrubbed"], default=None)
parser.add_argument('--training_balanced', type=str, help='balancing of training data used to train the model',
                    choices=["subsampled", "oversampled", "original"], default="original")
parser.add_argument('--seeds', '-s', default=0, nargs="+", type=int, help='the random seed the model was trained on')

args = parser.parse_args()

for num in args.seeds:
    print(num)
    load_path = f"checkpoints/bias_in_bios/roberta-base/finetuning/{args.training_data}/{args.training_balanced}/seed_{num}/best_model.pt"
    model_ = roBERTa_classifier(N_LABELS)
    load_checkpoint(model_, load_path)
    model = model_.roberta

    folder_ckpt = f"checkpoints/embedding_models/roberta-base/{args.training_data}/{args.training_balanced}/seed_{num}"
    Path(folder_ckpt).mkdir(parents=True, exist_ok=True)
    path_ckpt = folder_ckpt + "/model.pt"
    torch.save(model.state_dict(), path_ckpt)