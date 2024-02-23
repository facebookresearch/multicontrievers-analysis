import torch
from torch import nn
from transformers import RobertaModel
import torch.nn.functional as F


class roBERTa_classifier_PoE(nn.Module):
    def __init__(self, n_labels, n_biased_features=768):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base', add_pooling_layer=False, output_attentions=False,
                                                    output_hidden_states=False)
        self.classifier = nn.Linear(768, n_labels)
        self.n_biased_features = n_biased_features
        self.biased_classifier = nn.Linear(n_biased_features, n_labels)

    def forward(self, x, x_b, att_mask, att_mask_b, return_features=False):
        features = self.roberta(x, attention_mask=att_mask)[0]
        x = features[:, 0, :]
        # we do not want to update the LM based on the biased model's predictions

        if att_mask_b is not None:
            with torch.no_grad():
                features_b = self.roberta(x_b, attention_mask=att_mask_b)[0]
                x_b = features_b[:, 0, :].detach()

        # features_b = self.roberta(x_b, attention_mask=att_mask_b)[0]
        # x_b = features_b[:, 0, :]
        # x_b = grad_mul_const(x_b, 0.0)

        x = self.classifier(x)
        x_b = self.biased_classifier(x_b)
        if return_features:
            return x, features[:, 0, :], x_b, features_b[:, 0, :]
        else:
            return x, x_b


class POELoss(nn.Module):
    """Implements the product of expert loss."""
    """ Taken from https://github.com/rabeehk/robust-nli"""

    def __init__(self, size_average=True, ensemble_training=False, poe_alpha=1):
        super().__init__()
        self.size_average = size_average
        self.ensemble_training = ensemble_training
        self.poe_alpha = poe_alpha

    def compute_probs(self, inputs):
        prob_dist = F.softmax(inputs, dim=1)
        return prob_dist

    def forward(self, inputs, targets, inputs_adv, second_inputs_adv=None):
        targets = targets.view(-1, 1)
        pt = self.compute_probs(inputs)
        pt_adv = self.compute_probs(inputs_adv)
        if self.ensemble_training:
            pt_adv_second = self.compute_probs(second_inputs_adv)
            joint_pt = F.softmax((torch.log(pt) + torch.log(pt_adv) + torch.log(pt_adv_second)), dim=1)
        else:
            joint_pt = F.softmax((torch.log(pt) + self.poe_alpha * torch.log(pt_adv)), dim=1)
        joint_p = joint_pt.gather(1, targets)
        batch_loss = -torch.log(joint_p)

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss

# Multiplies the gradient of the given parameter by a constant.
class GradMulConst(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x,  const):
        ctx.const = const
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output*ctx.const, None

def grad_mul_const(x, const):
    return GradMulConst.apply(x, const)
