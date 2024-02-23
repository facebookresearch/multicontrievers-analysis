import torch
import torch.nn as nn
from transformers import RobertaModel


class roBERTa_classifier(nn.Module):
    def __init__(self, n_labels):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base', add_pooling_layer=False, output_attentions=False,
                                                    output_hidden_states=False)
        self.classifier = nn.Linear(768, n_labels)

    def forward(self, x, att_mask, return_features=False):
        features = self.roberta(x, attention_mask=att_mask)[0]
        x = features[:, 0, :]
        x = self.classifier(x)
        if return_features:
            return x, features[:, 0, :]
        else:
            return x



class GradReverse(torch.autograd.Function):
    """
    Extension of grad reverse layer
    """

    @staticmethod
    def forward(ctx, x, gradFactor):
        ctx.gradFactor = gradFactor
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.neg() * ctx.gradFactor
        return grad_output, None

    def grad_reverse(x, gradFactor):
        return GradReverse.apply(x, gradFactor)
