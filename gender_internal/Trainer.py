import abc
import json
from collections import Iterator, defaultdict
from abc import abstractmethod, ABC
from typing import NamedTuple

import numpy as np
import torch
import wandb
from allennlp.fairness import Independence, Separation, Sufficiency
from sklearn.neural_network import MLPClassifier
from torch.utils.data import DataLoader, Dataset
from pathlib import Path

from tqdm import tqdm

from .DFLModel import FocalLoss
from src.Data import Data, BiasInBiosData
from .Models import GradReverse
from .PoEModel import POELoss


class BestModel(NamedTuple):
    state_dict: dict
    epoch: int
    result: dict

def load_checkpoint(model, load_path):

    if load_path == None:
        return

    state_dict = torch.load(load_path)
    print(f'Model loaded from <== {load_path}')

    if callable(state_dict['model_state_dict']):
        model.load_state_dict(state_dict['model_state_dict']())
    else:
        model.load_state_dict(state_dict['model_state_dict'])

class Trainer(ABC):
    """
        A class abstracting the various tasks of training models.

        Provides methods at multiple levels of granularity:
        - Multiple epochs (fit)
        - Single epoch (train_epoch/test_epoch)
        - Single batch (train_batch/test_batch)
    """

    def __init__(self, model, loss_fn, optimizer, bsz, scheduler=None, device='cpu', seed=None):
        """
        Initialize the trainer.
        :param model: Instance of the model to train.
        :param loss_fn: The loss function to evaluate with.
        :param optimizer: The optimizer to train with.
        :param device: torch.device to run training on (CPU or GPU).
        """
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.verbose = False
        self.best_model: BestModel = None
        self.metric = 'acc'
        self.batch_size = bsz
        self.seed = seed

        # move to device
        self.model.to(self.device)

    def fit(self, train_data: Data, val_data: Data, num_epochs, checkpoint_folder, print_every=1,
            checkpoint_every=1):

        for epoch in range(1, num_epochs + 1):

            self.verbose = ((epoch % print_every) == 0 or (epoch == num_epochs - 1))
            self._print(f'--- EPOCH {epoch}/{num_epochs} ---', self.verbose)

            self.model.train()
            train_result = self.train_epoch(train_data)
            self.model.eval()
            valid_result = self.evaluate(val_data, "valid")

            self.best_checkpoint(valid_result, epoch)
            if (epoch % checkpoint_every) == 0:
                self.save_checkpoint(checkpoint_folder, epoch, valid_result, save_best=False)

            # print(train_result, valid_result)
            self.log_epoch_results(train_result, valid_result)

        self.save_checkpoint(checkpoint_folder, None, None, save_best=True)

    @abstractmethod
    def train_epoch(self, train_data: Data):
        """
                Train once over a training set (single epoch).
                :param train_data: the training data object
                :return: An epoch result dictionary.
                """
        ...

    def train_batch(self, batch):

        self.optimizer.zero_grad()
        res = self.forward_fn(batch)
        loss = res['loss']
        loss.backward()
        # grads = []
        # for p in self.model.parameters():
        #     grads.append(torch.norm(p.grad).item())
        # wandb.log({"avg grad": np.mean(grads), "batch loss": loss.item()})
        self.optimizer.step()

        if self.scheduler:
            self.scheduler.step()
            # lr = self.scheduler.get_last_lr()
            # print(lr)

        return res

    def save_checkpoint(self, save_folder, epoch, valid_result, save_best):

        if save_folder == None:
            return

        Path(save_folder).mkdir(parents=True, exist_ok=True)

        if not save_best:
            save_path = f"{save_folder}/ckpt_epoc_{epoch}.pt"
            torch.save({'model_state_dict': self.model.state_dict(),
                        'epoch_data': valid_result,
                        'epoch': epoch}, save_path)
        else:
            save_path = f"{save_folder}/best_model.pt"
            torch.save({'model_state_dict': self.best_model.state_dict,
                        'epoch_data': self.best_model.result,
                        'epoch': self.best_model.epoch}, save_path)

        print(f'Model saved to ==> {save_path}')

    def load_checkpoint(self, load_path):

        if load_path == None:
            return

        state_dict = torch.load(load_path)
        print(f'Model loaded from <== {load_path}')

        if callable(state_dict['model_state_dict']):
            self.model.load_state_dict(state_dict['model_state_dict']())
        else:
            self.model.load_state_dict(state_dict['model_state_dict'])

    @staticmethod
    def _print(message, verbose=True):
        """ Simple wrapper around print to make it conditional """
        if verbose:
            print(message)

    def best_checkpoint(self, valid_result, epoch):
        if (not self.best_model) or self.best_model.result[self.metric] < valid_result[self.metric]:
            self.best_model = BestModel(self.model.state_dict(), epoch, valid_result.copy())
            wandb.run.summary["best_metric"] = valid_result[self.metric]
            wandb.run.summary["best_epoch"] = epoch
            wandb.run.summary["metric"] = self.metric

    @abstractmethod
    def forward_fn(self, batch):
        ...

    @abstractmethod
    def evaluate(self, data: Data, split_type: str):
        ...

    @abstractmethod
    def log_epoch_results(self, train_result, valid_result):
        ...


class BiasInBiosTrainer(Trainer):

    def log_epoch_results(self, train_result, valid_result):

        train_result_new = {}
        for k in train_result:
            train_result_new[f"train_{k}"] = train_result[k]
        valid_result_new = {}
        for k in valid_result:
            valid_result_new[f"valid_{k}"] = valid_result[k]

        wandb.log({**train_result_new, **valid_result_new})

    def evaluate(self, data: Data, split_type: str):
        dl = DataLoader(data.dataset, batch_size=self.batch_size, shuffle=False)

        y_pred, logits = self.predict(dl)
        y = data.dataset.tensors[1].to(self.device)
        loss = self.loss_fn(logits, y)
        total_correct = torch.sum(y == logits.argmax(dim=1)).item()

        total_examples = len(y)
        accuracy = total_correct / total_examples
        perc = self.get_perc(data, split_type)
        gap_res = self.gap(data, y_pred, split_type, perc)
        fairness = self.allennlp_metrics(data, y_pred, perc)

        return {"loss": loss, "acc": accuracy, **gap_res, **fairness}

    def predict(self, data: DataLoader):
        self.model.eval()
        all_y_pred = []
        all_logits = []

        with torch.no_grad():
            for batch in tqdm(data):
                logits = self.forward_fn(batch)['logits']
                y_pred = logits.argmax(dim=1)
                all_y_pred.append(y_pred)
                all_logits.append(logits)

        all_y_pred = torch.cat(all_y_pred, dim=0)
        all_logits = torch.cat(all_logits, dim=0)
        return all_y_pred, all_logits

    def train_epoch(self, train_data: Data):
        losses = []
        total_correct = 0
        total_examples = 0

        train_iter = DataLoader(train_data.dataset, batch_size=self.batch_size, shuffle=True)

        # tmp = 0
        for batch in tqdm(train_iter):
            # if tmp>5:
            #     break
            # tmp+=1
            y = batch[1]
            batch_res = self.train_batch(batch)
            losses.append(batch_res['loss'].item())
            total_correct += batch_res['n_correct']
            total_examples += len(y)

        accuracy = total_correct / total_examples
        loss = np.mean(losses)

        return {"loss": loss, "acc": accuracy}

    def get_perc(self, data, split_type):
        # Load percentage of women in every profession
        # if self.seed is not None:
        #     perc_dict = torch.load(f"../data/biasbios/perc_{split_type}-seed_{self.seed}")
        # else:
        #     perc_dict = torch.load(f"../data/biasbios/perc_{split_type}-seed_{wandb.config.seed}")
        perc_dict = data.perc

        label_to_code = data.get_label_to_code()
        perc = []
        for i, (profession, label) in enumerate(sorted(label_to_code.items(),  key=lambda item: item[1])):

            # if profession in perc_dict:
            perc.append(perc_dict[profession])

        return perc

    # def gap(self, data: BiasInBiosDataFinetuning, y_pred, split_type, perc):

    #     tpr_gap = []
    #     fpr_gap = []
    #     precision_gap = []
    #     F1_gap = []

    #     golden_y = data.dataset.tensors[1]


    #     z = data.z

    #     for label in torch.unique(golden_y):

    #         # code_to_label = dict(enumerate(data.cat.categories))
    #         # if code_to_label[label.item()] in data.perc:
    #         m_res = self.metrics_fn(y_pred, golden_y, z, label.item(), 'M')
    #         f_res = self.metrics_fn(y_pred, golden_y, z, label.item(), 'F')

    #         tpr_gap.append(f_res["tpr"] - m_res["tpr"])
    #         fpr_gap.append(f_res["fpr"] - m_res["fpr"])
    #         precision_gap.append(f_res["precision"] - m_res["precision"])
    #         F1_gap.append(f_res["f1_score"] - m_res["f1_score"])

    #     # wandb.run.summary["employment_percentage"] = perc
    #     return {"tpr_gap": tpr_gap, "pearson_tpr_gap": np.corrcoef(perc, tpr_gap)[0, 1],
    #             "fpr_gap": fpr_gap, "pearson_fpr_gap": np.corrcoef(perc, fpr_gap)[0, 1],
    #             "precision_gap": precision_gap, "pearson_precision_gap": np.corrcoef(perc, precision_gap)[0, 1],
    #             "F1_gap": F1_gap, "pearson_F1_gap": np.corrcoef(perc, F1_gap)[0, 1],
    #             "mean abs tpr gap": np.abs(tpr_gap).mean(),
    #             "mean abs fpr gap": np.abs(fpr_gap).mean(),
    #             "mean abs f1 gap": np.abs(F1_gap).mean(),
    #             "mean abs precision gap": np.abs(precision_gap).mean(),
    #             "perc": perc,
    #             }

    def metrics_fn(self, y_pred: torch.Tensor, golden_y, z, label: int, gender: str):
        assert (len(y_pred) == len(golden_y))

        tp_indices = np.logical_and(np.logical_and(z == gender, golden_y.cpu() == label),
                                    y_pred.cpu() == label).int()  # only correct predictions of this gender

        n_tp = torch.sum(tp_indices).item()
        pos_indices = np.logical_and(z == gender, golden_y.cpu() == label).int()
        n_pos = torch.sum(pos_indices).item()
        tpr = n_tp / n_pos

        fp_indices = np.logical_and(np.logical_and(z == gender, golden_y.cpu() != label), y_pred.cpu() == label).int()
        neg_indices = np.logical_and(z == gender, golden_y.cpu() != label).int()
        n_fp = torch.sum(fp_indices).item()
        n_neg = torch.sum(neg_indices).item()
        fpr = n_fp / n_neg

        n_total_examples = len(y_pred)
        precision = n_tp / n_total_examples

        if precision * tpr == 0:
            f1_score = 0
        else:
            f1_score = 2 * ((precision * tpr) / (precision + tpr))

        return {"tpr": tpr, "fpr": fpr, "precision": precision, "f1_score": f1_score}

    def allennlp_metrics(self, data: BiasInBiosData, y_pred, perc):

        z = data.z.copy()
        z[z == 'M'] = 0
        z[z == 'F'] = 1
        z = torch.Tensor(z.astype(int))
        y = data.dataset.tensors[1].cpu()
        y_pred = y_pred.cpu()

        independence = Independence(data.n_labels, 2)
        independence(y_pred, z)
        independence_score = independence.get_metric()

        separation = Separation(data.n_labels, 2)
        separation(y_pred, y, z)
        separation_score = separation.get_metric()

        sufficiency = Sufficiency(data.n_labels, 2, dist_metric="wasserstein")
        sufficiency(y_pred, y, z)
        sufficiency_score = sufficiency.get_metric()

        self.dictionary_torch_to_number(independence_score)
        self.dictionary_torch_to_number(separation_score)
        self.dictionary_torch_to_number(sufficiency_score)

        separation_gaps = [scores[0] - scores[1] for label, scores in sorted(separation_score.items())] # positive value - more separation for women
        # pearson_separation_gaps = np.corrcoef(perc, separation_gaps)[0, 1]
        sufficiency_gaps = [scores[0] - scores[1] for label, scores in sorted(sufficiency_score.items())]
        # pearson_sufficiency_gaps = np.corrcoef(perc, sufficiency_gaps)[0, 1]

        return {"independence": json.dumps(independence_score), "separation": json.dumps(separation_score), "sufficiency": json.dumps(sufficiency_score),
                "independence_sum": independence_score[0] + independence_score[1],
                "separation_gaps": separation_gaps,
                "sufficiency_gaps": sufficiency_gaps}

    def dictionary_torch_to_number(self, d: dict):
        for k, v in d.items():
            if isinstance(v, dict):
                self.dictionary_torch_to_number(v)
            else:
                d[k] = v.item()

class FinetuningBiasInBiosTrainer(BiasInBiosTrainer):

    def forward_fn(self, batch):
        batch = tuple(t.to(self.device) for t in batch)
        X, y, att_mask = batch

        logits = self.model.forward(X, att_mask)  # shape (batch_size, n_labels)
        loss = self.loss_fn(logits, y)
        n_correct = torch.sum(y == logits.argmax(dim=1)).item()

        return {'loss': loss, 'n_correct': n_correct, 'logits': logits}


class NoFinetuningBiasInBiosTrainer(BiasInBiosTrainer):

    def forward_fn(self, batch):
        batch = tuple(t.to(self.device) for t in batch)
        X, y = batch  # X shape (bsz, 768)

        logits = self.model.forward(X)  # shape (bsz, n_labels)
        loss = self.loss_fn(logits, y)
        n_correct = torch.sum(y == logits.argmax(dim=1)).item()

        return {'loss': loss, 'n_correct': n_correct, 'logits': logits}

class PoEBiasInBiosTrainer(BiasInBiosTrainer):

    def __init__(self, model, loss_fn, optimizer, bsz, biased_model, scheduler=None, device='cpu', seed=None):
        super().__init__(model, loss_fn, optimizer, bsz, scheduler, device, seed)
        self.poe_loss_fn = POELoss()
        self.biased_model = biased_model

    def forward_fn(self, batch):
        batch = tuple(t.to(self.device) for t in batch)

        if len(batch) == 5:
            X, y, att_mask, X_b, att_mask_b = batch
        else:
            X, y, att_mask, X_b = batch
            att_mask_b = None

        # logits, logits_b = self.model.forward(X, X_b, att_mask, att_mask_b)  # shape (batch_size, n_labels)
        logits = self.model.forward(X, att_mask)

        X_b_np = X_b.cpu().numpy()
        y_np = y.cpu().numpy()

        # just a hack to see if we are on train mode
        if self.model.training:
            self.biased_model.partial_fit(X_b_np, y_np, np.arange(28))
        logits_b = torch.tensor(self.biased_model.predict_proba(X.cpu().numpy())).to(self.device)

        loss = self.poe_loss_fn(logits, y, logits_b)
        n_correct = torch.sum(y == logits.argmax(dim=1)).item()

        return {'loss': loss, 'n_correct': n_correct, 'logits': logits}

# class DFLBiasInBiosTrainer(BiasInBiosTrainer):
#
#     def __init__(self, model, loss_fn, optimizer, bsz, scheduler=None, device='cpu', seed=None):
#         super().__init__(model, loss_fn, optimizer, bsz, scheduler, device, seed)
#         self.dfl_loss_fn = FocalLoss()
#
#     def forward_fn(self, batch):
#
#         batch = tuple(t.to(self.device) for t in batch)
#         X, y, att_mask, z = batch
#
#         logits, logits_b = self.model.forward(X, att_mask)  # shape (batch_size, n_labels)
#         loss = self.dfl_loss_fn(logits, y, logits_b, z)
#         n_correct = torch.sum(y == logits.argmax(dim=1)).item()
#
#         return {'loss': loss, 'n_correct': n_correct, 'logits': logits}

class DFLBiasInBiosTrainer(BiasInBiosTrainer):

    def __init__(self, model, loss_fn, optimizer, bsz, biased_model, gamma=2, scheduler=None, device='cpu', seed=None):
        super().__init__(model, loss_fn, optimizer, bsz, scheduler, device, seed)
        self.dfl_loss_fn = FocalLoss(gamma=gamma)
        self.biased_model = biased_model

    def forward_fn(self, batch, is_validation=False):

        batch = tuple(t.to(self.device) for t in batch)
        X, y, att_mask, z = batch


        logits, features = self.model.forward(X, att_mask, return_features=True)
        features_np = features.detach().cpu().numpy()
        z_np = z.cpu().numpy()

        # just a hack to see if we are on train mode
        if self.model.training:
            self.biased_model.partial_fit(features_np, z_np.ravel(), np.arange(2))
        logits_b = torch.tensor(self.biased_model.predict_proba(features_np)).to(self.device)

        loss = self.dfl_loss_fn(logits, y, logits_b, z)
        n_correct = torch.sum(y == logits.argmax(dim=1)).item()

        # temp - debugging
        wandb.log({'biased_model_acc': self.biased_model.score(features_np, z_np.ravel())})
        return {'loss': loss, 'n_correct': n_correct, 'logits': logits}

class DFL_hard_BiasInBiosTrainer(BiasInBiosTrainer):

    def __init__(self, model, loss_fn, optimizer, bsz, gamma=2, scheduler=None, device='cpu', seed=None):
        super().__init__(model, loss_fn, optimizer, bsz, scheduler, device, seed)
        self.dfl_loss_fn = FocalLoss(gamma=gamma)

    def forward_fn(self, batch):

        batch = tuple(t.to(self.device) for t in batch)
        X, y, att_mask, z, z_preds = batch

        logits = self.model.forward(X, att_mask)

        loss = self.dfl_loss_fn(logits, y, z_preds, z)
        n_correct = torch.sum(y == logits.argmax(dim=1)).item()

        return {'loss': loss, 'n_correct': n_correct, 'logits': logits}

class BiasInBiosLMTrainer(Trainer):

    def __init__(self, model, loss_fn, optimizer, bsz, tokenizer, scheduler=None, device='cpu'):
        super().__init__(model, loss_fn, optimizer, bsz, scheduler, device)
        self.tokenizer = tokenizer
        self.mask_token = tokenizer(tokenizer.mask_token, add_special_tokens=False, return_attention_mask=False)['input_ids'][0]
        self.a_an_tokens = np.array(tokenizer(" a an", add_special_tokens=False, return_attention_mask=False)['input_ids'])
        self.labels_dic = None

    def forward_fn(self, batch):
        batch = tuple(t.to(self.device) for t in batch)
        X, y, att_mask = batch[0], batch[1], batch[2]
        y[X != self.mask_token] = -100
        res = self.model.forward(X, attention_mask=att_mask, labels=y)
        return res

    def train_epoch(self, train_data: Data):
        losses = []

        train_iter = DataLoader(train_data.dataset, batch_size=self.batch_size, shuffle=True)
        self.get_labels(train_data)

        for batch in tqdm(train_iter):

            batch_res = self.train_batch(batch)
            losses.append(batch_res['loss'].item())

        loss = np.mean(losses)

        return {"loss": loss}

    # def evaluate(self, data: BiasInBiosDataLM, split_type: str):
    #     self.get_labels(data)

    #     losses = []
    #     all_scores2 = []
    #     all_scores3 = []
    #     all_scores4 = []
    #     all_predictions2 = []
    #     all_predictions3 = []
    #     all_predictions4 = []
    #     all_losses = []

    #     iter_with_lengths = DataLoader(data.test_dataset, batch_size=self.batch_size, shuffle=False)
    #     iter_with_true_lengths = DataLoader(data.dataset, batch_size=self.batch_size, shuffle=False)
    #     with torch.no_grad():
    #         # tmp = 0
    #         for batch_len, batch in tqdm(list(zip(iter_with_lengths, iter_with_true_lengths))):
    #             # if tmp > 100:
    #             #     break
    #             # tmp+=1

    #             x2, y2, masks2, x3, y3, masks3, x4, y4, masks4, true_length = batch_len
    #             loss = self.forward_fn(batch).loss.item()
    #             all_losses.append(loss)

    #             scores2, prediction2, loss2 = self.predict_len((x2, y2, masks2), 2)
    #             all_scores2.append(scores2)
    #             all_predictions2.append(prediction2)

    #             scores3, prediction3, loss3 = self.predict_len((x3, y3, masks3), 3)
    #             all_scores3.append(scores3)
    #             all_predictions3.append(prediction3)

    #             scores4, prediction4, loss4 = self.predict_len((x4, y4, masks4), 4)
    #             all_scores4.append(scores4)
    #             all_predictions4.append(prediction4)

    #         all_scores2, all_predictions2 = np.concatenate(all_scores2, axis=0), np.concatenate(all_predictions2,axis=0)
    #         all_scores3, all_predictions3 = np.concatenate(all_scores3, axis=0), np.concatenate(all_predictions3, axis=0)
    #         all_scores4, all_predictions4 = np.concatenate(all_scores4, axis=0), np.concatenate(all_predictions4, axis=0)

    #         res = self.predict_and_compute_metrics(all_predictions2, all_predictions3,
    #                                                all_predictions4, all_scores2, all_scores3, all_scores4, data, losses)

    #     gap_res = self.gap(data, res, split_type)
    #     return {'loss': np.mean(all_losses), 'acc': res['acc'], **gap_res}

    # def predict_and_compute_metrics(self, all_predictions2, all_predictions3,
    #                                 all_predictions4, all_scores2, all_scores3, all_scores4, data : BiasInBiosDataLM, losses):
    #     n_correct = 0
    #     total_examples = len(all_scores2)
    #     all_predictions = []
    #     t_pos = defaultdict(lambda: defaultdict(lambda: 0))
    #     f_pos = defaultdict(lambda: defaultdict(lambda: 0))
    #     z = data.z
    #     y2 = data.test_dataset.tensors[1]
    #     y3 = data.test_dataset.tensors[4]
    #     y4 = data.test_dataset.tensors[7]
    #     x2 = data.test_dataset.tensors[0]
    #     x3 = data.test_dataset.tensors[3]
    #     x4 = data.test_dataset.tensors[6]
    #     true_length = data.test_dataset.tensors[9]
    #     labels = data.numerical_labels.cpu().numpy()

    #     for i in range(len(all_scores2)):  # for each item
    #         # We do not use the scores of first token (a, an) because it is very high and therefore length 2
    #         # is always be higher

    #         m2 = self._harmonic_mean(2, all_scores2[i][1:])
    #         m3 = self._harmonic_mean(3, all_scores3[i][1:])
    #         m4 = self._harmonic_mean(4, all_scores4[i][1:])

    #         if m2 > m3 and m2 > m4:
    #             all_predictions.append(all_predictions2[i])
    #             if true_length[i] == 2 and np.sum(
    #                     all_predictions2[i] == y2[i][x2[i] == self.mask_token].cpu().numpy()) == 2:
    #                 # print("yay")
    #                 n_correct += 1
    #                 t_pos[labels[i]][z[i]] += 1
    #             else:
    #                 f_pos[labels[i]][z[i]] += 1
    #         elif m3 > m2 and m3 > m4:
    #             all_predictions.append(all_predictions3[i])
    #             if true_length[i] == 3 and np.sum(
    #                     all_predictions3[i] == y3[i][x3[i] == self.mask_token].cpu().numpy()) == 3:
    #                 n_correct += 1
    #                 t_pos[labels[i]][z[i]] += 1
    #             else:
    #                 f_pos[labels[i]][z[i]] += 1
    #         elif m4 > m2 and m4 > m3:
    #             all_predictions.append(all_predictions4[i])
    #             if true_length[i] == 4 and np.sum(
    #                     all_predictions4[i] == y4[i][x4[i] == self.mask_token].cpu().numpy()) == 4:
    #                 n_correct += 1
    #                 t_pos[labels[i]][z[i]] += 1
    #             else:
    #                 f_pos[labels[i]][z[i]] += 1

    #     acc = n_correct / total_examples
    #     return {"acc": acc, "predictions": all_predictions,
    #             "true_positive_dict": t_pos, "false_positive_dict": f_pos}

    # def gap(self, data : BiasInBiosDataLM, res_dict : dict, split_type):
    #     # Load percentage of women in every profession
    #     # perc_dict = torch.load(f"../data/biasbios/perc_{split_type}-seed_{wandb.config.seed}")
    #     perc_dict = data.perc
    #     perc = []
    #     tpr_gap = []
    #     fpr_gap = []
    #     precision_gap = []
    #     F1_gap = []

    #     z = data.z
    #     label_to_code = data.get_label_to_code()
    #     golden_y = data.numerical_labels

    #     for i, (profession, label) in enumerate(label_to_code.items()):
    #         m_res = self.metrics_fn(res_dict, golden_y, z, label, 'M')
    #         f_res = self.metrics_fn(res_dict, golden_y, z, label, 'F')

    #         tpr_gap.append(f_res["tpr"] - m_res["tpr"])
    #         fpr_gap.append(f_res["fpr"] - m_res["fpr"])
    #         precision_gap.append(f_res["precision"] - m_res["precision"])
    #         F1_gap.append(f_res["f1_score"] - m_res["f1_score"])
    #         # profession_ = profession.replace(" ", "_")
    #         perc.append(perc_dict[profession])

    #     return {"tpr_gap": tpr_gap, "pearson_tpr_gap": np.corrcoef(perc, tpr_gap)[0, 1],
    #             "fpr_gap": fpr_gap, "pearson_fpr_gap": np.corrcoef(perc, fpr_gap)[0, 1],
    #             "precision_gap": precision_gap, "pearson_precision_gap": np.corrcoef(perc, precision_gap)[0, 1],
    #             "F1_gap": F1_gap, "pearson_F1_gap": np.corrcoef(perc, F1_gap)[0, 1],
    #             "mean abs tpr gap": np.abs(tpr_gap).mean(),
    #             "mean abs fpr gap": np.abs(fpr_gap).mean(),
    #             "mean abs f1 gap": np.abs(F1_gap).mean(),
    #             "mean abs precision gap": np.abs(precision_gap).mean(),
    #             "perc": perc
    #             }

    # def metrics_fn(self, res_dict : dict, golden_y, z, label, gender):

    #     n_tp = res_dict["true_positive_dict"][label][gender]
    #     pos_indices = np.logical_and(z == gender, golden_y.cpu() == label)
    #     n_pos = torch.sum(pos_indices).item()
    #     tpr = n_tp / n_pos

    #     neg_indices = np.logical_and(z == gender, golden_y.cpu() != label)
    #     n_fp = res_dict["false_positive_dict"][label][gender]
    #     n_neg = torch.sum(neg_indices).item()
    #     fpr = n_fp / n_neg

    #     n_total_examples = len(golden_y)
    #     precision = n_tp / n_total_examples

    #     if precision * tpr == 0:
    #         f1_score = 0
    #     else:
    #         f1_score = 2 * ((precision * tpr) / (precision + tpr))

    #     return {"tpr": tpr, "fpr": fpr, "precision": precision, "f1_score": f1_score}

    # def log_epoch_results(self, train_result, valid_result):

    #     train_result["train_loss"] = train_result.pop("loss")
    #     valid_result["valid_acc"] = valid_result.pop("acc")
    #     valid_result["valid_loss"] = valid_result.pop("loss")
    #     # Fields I don't want to log
    #     train_result.pop("tpr_gap")
    #     train_result.pop("fpr_gap")
    #     train_result.pop("F1_gap")
    #     train_result.pop("precision_gap")
    #     wandb.log({**train_result, **valid_result})

    # def get_labels(self, data: Data):
    #     if self.labels_dic is not None:
    #         return

    #     self.labels_dic = {
    #         2: [],
    #         3: [],
    #         4: [],
    #     }

    #     labels_1 = []
    #     labels_2 = []
    #     labels_3 = []

    #     X = data.dataset.tensors[0]
    #     y = data.dataset.tensors[1]
    #     self._print("Getting labels...", verbose=self.verbose)
    #     for i in range(len(y)):
    #         tokens = y[i][X[i] == self.mask_token]
    #         if len(tokens) == 2:
    #             labels_1.append(tokens[1:].cpu().numpy())
    #         elif len(tokens) == 3:
    #             labels_2.append(tokens[1:].cpu().numpy())
    #         elif len(tokens) == 4:
    #             labels_3.append(tokens[1:].cpu().numpy())

    #     self.labels_dic[2] = np.unique(labels_1, axis=0)
    #     self.labels_dic[3] = np.unique(labels_2, axis=0)
    #     self.labels_dic[4] = np.unique(labels_3, axis=0)

    # def predict_len(self, batch, len):

    #     # Get logits
    #     x, y, masks = batch
    #     with torch.no_grad():
    #         res = self.forward_fn(batch)
    #     logits = res['logits']
    #     bsz = x.size(0)

    #     # Get only logits of masked tokens
    #     logits = logits[x == self.mask_token, :].reshape(bsz, len, -1)
    #     # Relevant tokens to look on their scores - only professions from the dataset
    #     labels = np.array(self.labels_dic[len])

    #     # get scores for a and an tokens
    #     a = logits[:, 0, :][:, self.a_an_tokens].max(dim=1).values.cpu()
    #     # get predicion - a or an
    #     a1 = self.a_an_tokens[logits[:, 0, :][:, self.a_an_tokens].max(dim=1).indices.cpu()]

    #     # get scores and predicition for first token of profession
    #     b = logits[:, 1, :][:, labels[:, 0]].max(dim=1).values.cpu()
    #     b1 = labels[:, 0][logits[:, 1, :][:, labels[:, 0]].max(dim=1).indices.cpu()]

    #     if len > 2:
    #         c = logits[:, 2, :][:, labels[:, 1]].max(dim=1).values.cpu()
    #         c1 = labels[:, 1][logits[:, 2, :][:, labels[:, 1]].max(dim=1).indices.cpu()]
    #     else:
    #         scores = np.stack((a, b), axis=1)
    #         prediction = np.stack((a1, b1), axis=1)
    #         return scores, prediction, res['loss'].item()
    #     if len > 3:
    #         d = logits[:, 3, :][:, labels[:, 2]].max(dim=1).values.cpu()
    #         d1 = labels[:, 2][logits[:, 3, :][:, labels[:, 2]].max(dim=1).indices.cpu()]
    #     else:
    #         scores = np.stack((a, b, c), axis=1)
    #         prediction = np.stack((a1, b1, c1), axis=1)
    #         return scores, prediction, res['loss'].item()

    #     # Len = 4
    #     scores = np.stack((a, b, c, d), axis=1)
    #     prediction = np.stack((a1, b1, c1, d1), axis=1)
    #     return scores, prediction, res['loss'].item()

    # @staticmethod
    # def _harmonic_mean(n, scores):
    #     return n / ((1 / torch.Tensor(scores)).sum())


class BiasInBiosAdversarialTrainer(Trainer):

    def __init__(self, model, detector, loss_fn, detector_loss_fn, optimizer, detector_optimizer, bsz, scheduler=None, device='cpu'):
        super().__init__(model, loss_fn, optimizer, bsz, scheduler, device)
        self.detector = detector
        self.detector_optimizer = detector_optimizer
        self.detector_loss_fn = detector_loss_fn
        self.detector.to(device)

    def train_epoch(self, train_data: Data):
        losses = []
        detector_losses = []
        total_correct = 0
        detector_total_correct = 0
        total_examples = 0

        train_iter = DataLoader(train_data.dataset, batch_size=self.batch_size, shuffle=True)

        k = 0
        tmp = 0
        for batch in tqdm(train_iter):
            if tmp > 50:
                break
            tmp+=1

            y = batch[1]
            batch_res = self.train_batch(batch)

            losses.append(batch_res['main_loss'].item())
            detector_losses.append(batch_res['detector_loss'].item())

            total_correct += batch_res['n_correct']
            detector_total_correct += batch_res['n_correct_detector']
            total_examples += len(y)

            if k % 10 == 0:
                print()
                print(f"loss:{batch_res['main_loss'].item()}, acc: {batch_res['n_correct'] / len(y)}, detector_loss: {batch_res['detector_loss'].item()}, detector_acc: {batch_res['n_correct_detector'] / len(y)}")
            k+=1

        self.save_checkpoint(f"checkpoints/bias_in_bios/roberta-base/finetuning_adversarial/raw/original/seed_0", None, None, save_best=False)
        accuracy = total_correct / total_examples
        detector_accuracy = detector_total_correct / total_examples
        loss = np.mean(losses)
        detector_loss = np.mean(detector_losses)

        #tmp
        return {"loss": loss, "acc": accuracy, "detector_loss": detector_loss, "detector_accuracy": detector_accuracy}

    def train_batch(self, batch):

        self.optimizer.zero_grad()
        res = self.forward_fn(batch)
        main_loss = res['main_loss']
        detector_loss = res['detector_loss']
        main_loss.backward(retain_graph=True)
        detector_loss.backward(retain_graph=False)
        self.optimizer.step()
        self.detector_optimizer.step()

        if self.scheduler:
            self.scheduler.step()

        return res

    def forward_fn(self, batch):
        batch = tuple(t.to(self.device) for t in batch)
        X, y, att_mask, z = batch

        logits, features = self.model.forward(X, att_mask, return_features=True)  # shape (batch_size, n_labels)
        main_loss = self.loss_fn(logits, y)

        gradFactor = 0.1 # tmp
        input_features = GradReverse.grad_reverse(features, gradFactor)
        z_logits = self.detector.forward(input_features)
        detector_loss = self.detector_loss_fn(z_logits, z)

        n_correct = torch.sum(y == logits.argmax(dim=1)).item()
        n_correct_detector = torch.sum(z == z_logits.argmax(dim=1)).item()

        return {'main_loss': main_loss, 'n_correct': n_correct, 'logits': logits, 'z_logits': z_logits,
                'detector_loss': detector_loss, 'n_correct_detector': n_correct_detector}

    def evaluate(self, data: Data, split_type: str):
        dl = DataLoader(data.dataset, batch_size=self.batch_size, shuffle=False)

        y_pred, logits, z_pred, detector_logits = self.predict(dl)

        y = data.dataset.tensors[1].to(self.device)
        loss = self.loss_fn(logits, y)
        total_correct = torch.sum(y == logits.argmax(dim=1)).item()

        z = data.dataset.tensors[3]
        detector_loss = self.loss_fn(detector_logits, z)
        total_correct_detector = torch.sum(z == detector_logits.argmax(dim=1)).item()

        total_examples = len(y)
        accuracy = total_correct / total_examples
        detector_accuracy = total_correct_detector / total_examples
        gap_res = self.gap(data, y_pred, split_type)

        return {"loss": loss, "acc": accuracy,
                "detector_loss": detector_loss, "detector_acc": detector_accuracy,
                **gap_res}

    def predict(self, data: DataLoader):
        self.model.eval()
        all_y_pred = []
        all_z_pred = []
        all_logits = []
        all_detector_logits = []

        with torch.no_grad():
            for batch in tqdm(data):
                logits, z_logits = self.forward_fn(batch)['logits'], self.forward_fn(batch)['z_logits']
                y_pred = logits.argmax(dim=1)
                z_pred = z_logits.argmax(dim=1)

                all_y_pred.append(y_pred)
                all_z_pred.append(z_pred)

                all_logits.append(logits)
                all_detector_logits.append(z_logits)

        all_y_pred = torch.cat(all_y_pred, dim=0)
        all_logits = torch.cat(all_logits, dim=0)
        all_detector_logits = torch.cat(all_detector_logits, dim=0)

        return all_y_pred, all_logits, all_z_pred, all_detector_logits

    def log_epoch_results(self, train_result, valid_result):
        train_result["train_acc"] = train_result.pop("acc")
        train_result["train_loss"] = train_result.pop("loss")
        valid_result["valid_acc"] = valid_result.pop("acc")
        valid_result["valid_loss"] = valid_result.pop("loss")
        # Fields I don't want to log
        valid_result.pop("tpr_gap")
        valid_result.pop("fpr_gap")
        valid_result.pop("F1_gap")
        valid_result.pop("precision_gap")
        wandb.log({**train_result, **valid_result})