import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
import numpy as np

# ------------------------------------------------------------
# Multilabel Loss Functions
# ------------------------------------------------------------

class BCELoss(nn.Module):
    """
    Binary Cross-Entropy loss for multi-label classification.
    Applies a sigmoid activation and then BCELoss.
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.criterion = nn.BCELoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        return self.criterion(probs, targets)


class ASLLoss(nn.Module):
    """
    Asymmetric Loss (ASL) for multi-label classification.
    Reference:
      Ridnik et al., "Asymmetric Loss For Multi-Label Classification", ICCV 2021.

    Args:
        gamma_neg (float): focusing parameter for negative targets.
        gamma_pos (float): focusing parameter for positive targets.
        clip (float): clip value for negative probabilities (to avoid easy negatives dominating).
        eps (float): small epsilon to avoid log(0).
        disable_torch_grad_focal_loss (bool): if True, disable gradient for focal loss weight computation.
    """
    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 1.0,
        clip: float = 0.05,
        eps: float = 1e-8,
        disable_torch_grad_focal_loss: bool = False,
        **kwargs
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.disable_grad = disable_torch_grad_focal_loss

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # probabilities for positive and negative
        prob_pos = torch.sigmoid(logits)
        prob_neg = 1.0 - prob_pos

        # asymmetric clipping
        if self.clip and self.clip > 0:
            prob_neg = (prob_neg + self.clip).clamp(max=1.0)

        # basic cross-entropy term
        loss = targets * torch.log(prob_pos.clamp(min=self.eps))
        loss += (1 - targets) * torch.log(prob_neg.clamp(min=self.eps))

        # asymmetric focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_grad:
                with torch.no_grad():
                    weight = torch.pow(
                        1 - prob_pos * targets - prob_neg * (1 - targets),
                        self.gamma_pos * targets + self.gamma_neg * (1 - targets)
                    )
            else:
                weight = torch.pow(
                    1 - prob_pos * targets - prob_neg * (1 - targets),
                    self.gamma_pos * targets + self.gamma_neg * (1 - targets)
                )
            loss = loss * weight

        return -loss.sum()


# ------------------------------------------------------------
# Multiclass Loss Functions
# ------------------------------------------------------------

class CELoss(nn.Module):
    """
    Standard cross-entropy loss for multi-class classification.
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.criterion(logits, targets)


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification.
    Reduces the relative loss for well-classified examples to focus training on hard examples.

    Args:
        gamma (float): focusing parameter.
    """
    def __init__(self, gamma: float = 2.0, **kwargs):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # one-hot encode targets
        num_classes = logits.size(1)
        targets_one_hot = F.one_hot(targets, num_classes).float()

        prob = torch.sigmoid(logits)
        # focal weight
        weight = (1 - prob) * targets_one_hot + prob * (1 - targets_one_hot)
        weight = weight.pow(self.gamma)

        bce = F.binary_cross_entropy_with_logits(logits, targets_one_hot, reduction='none')
        return (weight * bce).sum()


class BalancedSoftmaxCELoss(nn.Module):
    """
    Balanced Softmax Cross-Entropy Loss for handling class imbalance.

    Args:
        labels (Sequence[int]): training labels to compute class frequencies.
    """
    def __init__(self, labels, **kwargs):
        super().__init__()
        label_count = Counter(labels)
        num_classes = len(label_count)
        freqs = np.zeros(num_classes, dtype=np.float32)
        for label, count in label_count.items():
            freqs[label] = count
        # register as buffer for device-awareness
        log_prior = np.log(freqs)
        self.register_buffer('log_prior', torch.tensor(log_prior))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # adjust logits by log-prior
        adjusted = logits + self.log_prior.unsqueeze(0)
        return F.cross_entropy(adjusted, targets)
