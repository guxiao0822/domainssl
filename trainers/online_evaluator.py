import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from copy import deepcopy
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.preprocessing import label_binarize
from sklearn.neighbors import KNeighborsClassifier
from losses import *


def compute_metrics(y_pred: np.ndarray,
                    y_true: np.ndarray,
                    multiclass: bool,
                    metric: str = 'auc') -> np.ndarray:
    """
    Compute per-class AUC or F1 scores.
    """
    # One-hot encode if multiclass
    if multiclass:
        y_true_oh = label_binarize(y_true, classes=np.arange(y_pred.shape[1]))
    else:
        y_true_oh = y_true

    scores = []
    C = y_pred.shape[1]
    for c in range(C):
        true_c = y_true_oh[:, c]
        pred_c = y_pred[:, c]
        if metric == 'auc':
            # skip classes without variation
            if true_c.sum() > 0 and true_c.sum() < len(true_c):
                scores.append(roc_auc_score(true_c, pred_c))
            else:
                scores.append(np.nan)
        else:  # f1
            if multiclass:
                preds = y_pred.argmax(axis=1)
                return f1_score(y_true, preds, average=None)
            else:
                preds = (pred_c >= 0.5).astype(int)
                scores.append(f1_score(true_c, preds, zero_division=0))
    return np.array(scores)


class OnlineEvaluator:
    """
    Evaluates a model via linear probe, fine-tuning, or k-NN.
    """
    LOSS_MAP = {
        'CE': CELoss,
        'BCE': BCELoss,
        'Focal': FocalLoss,
        'ASL': ASLLoss,
        'BSCE': BalancedSoftmaxCELoss,
    }

    def __init__(self,
                 mode: str = 'linear',
                 epochs: int = 20,
                 device: torch.device = None,
                 multiclass: bool = False,
                 num_classes: int = 4,
                 metric: str = 'auc',
                 loss: str = 'BSCE',
                 **kwargs
                 ):
        self.mode = mode
        self.epochs = epochs
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.multiclass = multiclass
        self.num_classes = num_classes
        self.metric = metric.lower()
        self.loss_name = loss
        self.best_score = 0.0

    def _get_loss(self, labels):
        if self.loss_name not in self.LOSS_MAP:
            raise ValueError(f"Unknown loss '{self.loss_name}'")
        LossCls = self.LOSS_MAP[self.loss_name]
        return LossCls(labels=labels).to(self.device)

    def _extract(self, backbone, loader):
        feats, labs = [], []
        backbone.eval()
        with torch.no_grad():
            for batch in loader:
                x = batch['x'].to(self.device).float()
                # handle multi-view: reshape
                if x.ndim == 4:
                    B, V, C, L = x.shape
                    x = x.view(B * V, C, L)
                feats.append(backbone(x).cpu().numpy())
                labs.append(batch['y'].cpu().numpy())
        return np.vstack(feats), np.hstack(labs)

    def online_train(self, model, train_loader, val_loader):
        # prepare
        self.criterion = self._get_loss(train_loader.dataset.labels)
        backbone = deepcopy(model.encoder).to(self.device)
        head = Linear(model.classifier.in_features, self.num_classes).to(self.device)

        # extract features
        X_train, y_train = self._extract(backbone, train_loader)
        X_val, y_val = self._extract(backbone, val_loader)

        if self.mode in ('linear', 'ft'):
            params = head.parameters() if self.mode == 'linear' else list(backbone.parameters()) + list(head.parameters())
            opt = torch.optim.Adam(params, lr=1e-3 if self.mode=='linear' else 1e-4, weight_decay=1e-4)
            # train
            head.train()
            if self.mode == 'ft': backbone.train()
            for _ in range(self.epochs):
                opt.zero_grad()
                logits = head(torch.from_numpy(X_train).to(self.device))
                loss = self.criterion(logits, torch.from_numpy(y_train).to(self.device))
                loss.backward(); opt.step()
            # eval
            logits_val = head(torch.from_numpy(X_val).to(self.device)).cpu().data.numpy()
        else:  # knn
            knn = KNeighborsClassifier(n_neighbors=10)
            knn.fit(X_train, y_train)
            logits_val = knn.predict_proba(X_val)

        scores = compute_metrics(logits_val, y_val, self.multiclass, self.metric)
        score = np.nanmean(scores)
        if score > self.best_score:
            self.best_score = score

        print(f"\033[92mBest {self.metric.upper()} Score: {score:.3f} Best Score: {self.best_score:.3f}\033[0m")

        return score

    def evaluate(self, model, loader):
        # only for linear/ft
        backbone = model.backbone.to(self.device)
        head = Linear(model.fc.in_features, self.num_classes).to(self.device)
        X, y = self._extract(backbone, loader)
        logits = head(torch.from_numpy(X).to(self.device)).cpu().numpy()
        return compute_metrics(logits, y, self.multiclass, self.metric)