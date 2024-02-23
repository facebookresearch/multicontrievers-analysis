import torch
from torch import nn
from transformers import RobertaModel
import torch.nn.functional as F


class roBERTa_classifier_DFL(nn.Module):
    def __init__(self, n_labels, n_protected_labels):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base', add_pooling_layer=False, output_attentions=False,
                                                    output_hidden_states=False)
        self.classifier = nn.Linear(768, n_labels)
        self.biased_classifier = nn.Linear(768, n_protected_labels)

    def forward(self, x, att_mask, return_features=False):
        features = self.roberta(x, attention_mask=att_mask)[0]
        x = features[:, 0, :]
        # we do not want to update the LM based on the biased model's predictions
        x_b = x.detach()

        x = self.classifier(x)
        x_b = self.biased_classifier(x_b)
        if return_features:
            return x, features[:, 0, :], x_b
        else:
            return x, x_b

""" Taken from https://github.com/rabeehk/robust-nli"""
class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, size_average=True, ensemble_training=False, aggregate_ensemble="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.size_average = size_average
        self.ensemble_training = ensemble_training
        self.aggregate_ensemble=aggregate_ensemble

    def compute_probs(self, inputs, targets):
        prob_dist = F.softmax(inputs, dim=1)
        pt = prob_dist.gather(1, targets)
        return pt

    def aggregate(self, p1, p2, operation):
        if self.aggregate_ensemble == "mean":
            result = (p1+p2)/2
            return result
        elif self.aggregate_ensemble == "multiply":
            result = p1*p2
            return result
        else:
            assert NotImplementedError("Operation ", operation, "is not implemented.")

    def forward(self, inputs, targets, inputs_adv, targets_adv, second_inputs_adv=None):
        targets = targets.view(-1, 1)
        # norm = 0.0
        pt = self.compute_probs(inputs, targets)
        pt_scale = self.compute_probs(inputs_adv, targets_adv)
        # print(inputs_adv, pt_scale)
        if self.ensemble_training:
            pt_scale_second = self.compute_probs(second_inputs_adv, targets)
            if self.aggregate_ensemble in ["mean", "multiply"]:
                pt_scale_total = self.aggregate(pt_scale, pt_scale_second, "mean")
                batch_loss = -self.alpha * (torch.pow((1 - pt_scale_total), self.gamma)) * torch.log(pt)
        else:
            batch_loss = -self.alpha * (torch.pow((1 - pt_scale), self.gamma)) * torch.log(pt)
        # norm += self.alpha * (torch.pow((1 - pt_scale), self.gamma))

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()

        return loss