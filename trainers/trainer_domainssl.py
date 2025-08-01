import os
import time
from typing import Tuple
from copy import deepcopy
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
from einops import rearrange
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, average_precision_score

from utils.utils import save_checkpoint, AverageMeter, ProgressMeter


torch.manual_seed(0)


class Prototype(nn.Module):
    """
    Prototype representation: maintains moving-average class means.
    """
    def __init__(self, feat_dim: int, num_class: int, beta: float = 0.5):
        super().__init__()
        self.num_class = num_class
        self.beta = beta
        self.mean = torch.zeros(num_class, feat_dim)

    def update_statistics(self, feats: torch.Tensor, labels: torch.Tensor, epsilon: float = 1e-5):
        """
        Update per-class prototype means using batch features.

        Args:
            feats: Tensor of shape (batch_size, feat_dim)
            labels: LongTensor of shape (batch_size,), values in [0, num_class-1]
            epsilon: small constant to avoid division by zero
        """
        # feats: (B, D), labels: (B,)
        B, D = feats.shape
        C = self.num_class
        device = feats.device

        # Sum features per class
        sum_feats = torch.zeros(C, D, device=device)
        sum_feats.index_add_(0, labels, feats)

        # Count samples per class
        counts = torch.bincount(labels, minlength=C).unsqueeze(1).clamp(min=epsilon)

        # Compute batch means for each class
        batch_mean = sum_feats / counts  # shape: (C, D)

        # Determine which classes are present in the batch
        present = (counts.squeeze() > 0).float().unsqueeze(1)  # shape: (C, 1)

        # Moving-average update of prototype means
        self.mean = self.mean.to(device)
        self.mean = (self.mean * (1 - present)) + \
                    ((self.mean * self.beta + batch_mean * (1 - self.beta)) * present)

    def freeze(self):
        """
        Detach prototype means.
        """
        self.mean.detach_()


class DomainSSLTrainer:
    """
     Trainer combining instance- and cluster-level contrastive learning with prototypes.

     Args:
         config: dict with training settings
         optimizer: torch optimizer
         scheduler: learning-rate scheduler
         model: SimCLR-like model
     """

    def __init__(
            self,
            config: dict,
            optimizer: torch.optim.Optimizer,
            scheduler,
            model: nn.Module
    ):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epochs = config['training']['epochs']
        self.batch_size = config['training']['batch_size']
        self.temp = config['training']['temp']
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.best_score = 0.0
        self.exp_name = config['exp']
        self.n_views = 2
        # prototype dims from model
        feat_dim = model.projection_head[-1].out_features
        num_class = config['training']['num_proto']
        self.proto = Prototype(feat_dim, num_class).to(self.device)
        self.scaler = GradScaler()

    def exclude_topk(self, feats: torch.Tensor, topk: int = 1) -> torch.BoolTensor:
        """
        Build mask excluding top-k nearest neighbors per instance.
        """
        B = feats.size(0)
        dist = torch.cdist(feats, feats)
        # ignore self
        idx = torch.arange(B, device=dist.device)
        dist[idx, idx] = float('inf')
        # topk indices
        _, knn = dist.topk(topk, largest=False)
        mask = torch.zeros_like(dist, dtype=torch.bool)
        mask.scatter_(1, knn, True)
        # replicate for two views
        return mask.repeat(self.n_views, self.n_views)

    def info_nce_loss(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute InfoNCE loss logits and labels.
        """
        batch_size = features.size(0) // self.n_views
        labels = torch.cat([torch.arange(batch_size) for _ in range(self.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float().to(self.device)
        features = F.normalize(features, dim=1)
        similarity = features @ features.T
        mask = torch.eye(labels.size(0), device=self.device).bool()
        labels = labels[~mask].view(labels.size(0), -1)
        similarity = similarity[~mask].view(similarity.size(0), -1)
        positives = similarity[labels.bool()].view(labels.size(0), -1)
        negatives = similarity[~labels.bool()].view(similarity.size(0), -1)
        logits = torch.cat([positives, negatives], dim=1) / self.temp
        targets = torch.zeros(logits.size(0), dtype=torch.long, device=self.device)
        return logits, targets


    def info_nce_loss_wmask(self, features, exclude_mask):
        batch_size = features.shape[0]/2
        labels = torch.cat([torch.arange(batch_size) for i in range(self.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.device)

        features = F.normalize(features, dim=1)

        logits = torch.matmul(features, features.T)/self.temp

        # Prepare mask to remove self-similarities and apply exclude_mask
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)

        # Consider exclude_mask as indicating additional positives
        # pos_mask = exclude_mask.to(self.device)
        pos_mask = exclude_mask.float().to(self.device) + labels * ~mask

        # compute log_prob
        exp_logits = torch.exp(logits) * ~mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point.
        # Edge case e.g.:-
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan]
        mask_pos_pairs = pos_mask.sum(1)
        # mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (pos_mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = -mean_log_prob_pos
        loss = loss.mean()

        return loss # , labels

    def compute_dynamic_tau(
            self,
            feats: torch.Tensor,
            labels: torch.Tensor,
            m: float = 10.0
    ) -> torch.Tensor:
        """
        Compute per-cluster temperature τ_k according to Eq. (5):
            τ_k = ( sum_i ||z_i - c_k||_2 ) / (n_k * log(n_k + m) )
        then rescale so mean(τ) == self.temp.

        Args:
            feats:    (B, D)  features (one view only!)
            labels:   (B,)    cluster labels in [0..C-1]
            m:        smoothing constant (default 10)

        Returns:
            tau:      (C,)    per-cluster temperatures
        """
        #   normalize features
        feats = F.normalize(feats, dim=1)
        #   normalize prototype centers
        centers = F.normalize(self.proto.mean, dim=1)

        C, D = self.proto.mean.shape
        device = feats.device

        # compute n_k and sum of distances for each cluster
        tau = feats.new_zeros(C)
        counts = torch.bincount(labels, minlength=C).to(device)
        for k in range(C):
            n_k = counts[k].item()
            if n_k > 0:
                # compute L2 distances of all z_i with label k to center c_k
                d = (feats[labels == k] - centers[k]).norm(dim=1).sum()
                tau[k] = d / (n_k * torch.log(torch.tensor(n_k + m, device=device)))
            else:
                tau[k] = 0.0

        # fill zeros, renormalize mean→self.temp as before…
        nonzero = tau > 0
        if nonzero.any():
            mean_nonzero = tau[nonzero].mean()
            tau[~nonzero] = mean_nonzero

        tau = tau * (self.temp / tau.mean().clamp(min=1e-6))

        return tau

    def train_one_epoch(self, loader: DataLoader, epoch: int) -> float:
        """Run one epoch of training."""
        self.model.train()
        loss_meter = AverageMeter('Loss', ':6.4f')
        proto_meter = AverageMeter('Proto', ':6.4f')
        prog = ProgressMeter(len(loader), [loss_meter, proto_meter], prefix=f'Epoch [{epoch}]')

        total_loss = 0.0
        self.proto.freeze()

        for i, batch in enumerate(loader):
            f = batch['f'].to(self.device)
            labels = batch['c'].to(self.device).long().squeeze()
            x = rearrange(batch['x'].to(self.device).float(), 'b v c t -> (v b) c t')

            # instance mask
            pos_mask = self.exclude_topk(f)
            # forward
            feats = self.model(x)
            loss_ins = self.info_nce_loss_wmask(feats, pos_mask)

            # prototype loss
            self.proto.update_statistics(feats, labels.repeat(self.n_views))

            # compute τ
            labels_all = labels.repeat(self.n_views)  # shape (B * v,)
            tau = self.compute_dynamic_tau(feats, labels_all, m=10.0)  # returns (C,)

            proto_norm = F.normalize(self.proto.mean, dim=1)
            feat_norm = F.normalize(feats, dim=1)
            scores = feat_norm @ proto_norm.T
            if epoch > 1:
                scores = scores / tau.unsqueeze(0)  # broadcast per-cluster τ_k
            else:
                scores = scores / self.temp
            loss_proto = F.cross_entropy(scores, labels.repeat(self.n_views))

            loss = loss_ins + min(1, max(epoch-40,0) / 1000) * loss_proto

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            loss_meter.update(loss_ins.item(), x.size(0))
            proto_meter.update(loss_proto.item(), x.size(0))
            self.proto.freeze()

            if i % 10 == 0:
                prog.display(i)

        return total_loss / len(loader)

    def validate(self, val_loader) -> float:
        """Validate for one epoch."""
        self.model.eval()
        acc_meter = AverageMeter('Val Acc', ':6.2f')
        with torch.no_grad():
            for batch in val_loader:
                x = batch['x'].to(self.device).float()
                x = rearrange(x, 'b n c t -> (n b) c t')
                features = self.model(x)
                logits, targets = self.info_nce_loss(features)
                preds = logits.argmax(dim=1)
                acc = (preds == targets).float().mean().item()
                acc_meter.update(acc, x.size(0))
        print(f'Validation Accuracy: {acc_meter.avg:.4f}')
        return acc_meter.avg

    def train(
            self,
            train_loader: DataLoader,
            val_loader: DataLoader,
            ds_train_loader: DataLoader,
            ds_val_loader: DataLoader,
            online_evaluator
    ) -> None:
        """Full training with early stopping via online evaluation."""
        os.makedirs(f'exp_log/{self.exp_name}', exist_ok=True)
        patience, no_improve = 5, 0

        for epoch in range(1, self.epochs + 1):
            train_loss = self.train_one_epoch(train_loader, epoch)
            if epoch % 10 == 0:
                score = online_evaluator.online_train(self.model, ds_train_loader, ds_val_loader)
                if score > self.best_score:
                    save_checkpoint(self.model.encoder, self.optimizer, epoch, train_loss,
                                    f'exp_log/{self.exp_name}/best.pth')
                    self.best_score = score
                    no_improve = 0
                else:
                    no_improve += 1
                if no_improve >= patience:
                    print(f'Early stopping at epoch {epoch}')
                    break
            self.validate(val_loader)

        save_checkpoint(self.model.encoder, self.optimizer, epoch, train_loss,
                        f'exp_log/{self.exp_name}/last.pth')
