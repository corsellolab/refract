"""Trainers for ranking models"""
import logging

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from refract.losses import lambdaLoss

logger = logging.getLogger(__name__)


class NNRankerTrainer:
    def __init__(self, model, optimizer, num_workers=4, device=None):
        self.model = model
        self.optimizer = optimizer
        self.num_workers = num_workers
        self.writer = SummaryWriter()
        self.splitter = KFold(n_splits=5, shuffle=True, random_state=42)
        if device:
            self.device = device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def eval(self, dataset):
        preds = []
        trues = []
        model = self.model
        model.eval()
        dl = DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=self.num_workers
        )
        for ex in dl:
            feat, labels = ex
            pred = model(feat.to(self.device))[0]
            preds.append(float(pred))
            trues.append(float(labels[0]))
        # compute pearsonr
        corr = pearsonr(preds, trues)
        return preds, trues, corr

    def train(self, dataset, batch_size, epochs=200):
        """Train model for a given number of epochs"""
        model = self.model
        model.to(self.device)
        model.train()
        dl = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=self.num_workers
        )
        for epoch in range(epochs):
            for batch in dl:
                features, labels = batch
                features = features.to(self.device)
                labels = labels.to(self.device)
                pred_scores = model(features).squeeze()
                loss = lambdaLoss(
                    pred_scores,
                    labels,
                    weighing_scheme="lambdaRank_scheme",
                    k=self.SLATE_LENGTH,
                )
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            logger.info("Epoch: {}, Loss: {}".format(epoch, loss.item()))  # type: ignore
            self.writer.add_scalar("Loss/train", loss.item(), epoch)  # type: ignore

        _, _, train_pearsonr = self.eval(dataset)
        self.writer.add_scalar("Pearsonr/train", train_pearsonr[0])  # type: ignore
