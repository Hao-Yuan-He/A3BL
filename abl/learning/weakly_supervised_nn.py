# coding: utf-8

import torch
import numpy
from torch.utils.data import DataLoader
from ..utils.logger import print_log
from ..dataset import ClassificationDataset, WeaklyClassificationDataset
from .basic_nn import BasicNN
import os
from typing import List, Any, T, Optional, Callable, Tuple


class WeaklySupervisedNN(BasicNN):
    """ 
    Wrap NN models into the form of an sklearn estimator.
    Implemented with Weakly Supervised Perspective

    Parameters
    ----------
    model : torch.nn.Module
        The PyTorch model to be trained or used for prediction.
    criterion : torch.nn.Module
        The loss function used for training.
    optimizer : torch.optim.Optimizer
        The optimizer used for training.
    device : torch.device, optional
        The device on which the model will be trained or used for prediction, by default torch.device("cpu").
    batch_size : int, optional
        The batch size used for training, by default 32.
    num_epochs : int, optional
        The number of epochs used for training, by default 1.
    stop_loss : Optional[float], optional
        The loss value at which to stop training, by default 0.01.
    num_workers : int
        The number of workers used for loading data, by default 0.
    save_interval : Optional[int], optional
        The interval at which to save the model during training, by default None.
    save_dir : Optional[str], optional
        The directory in which to save the model during training, by default None.
    transform : Callable[..., Any], optional
        A function/transform that takes in an object and returns a transformed version, by default None.
    collate_fn : Callable[[List[T]], Any], optional
        The function used to collate data, by default None.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device = torch.device("cpu"),
        batch_size: int = 32,
        num_epochs: int = 1,
        stop_loss: Optional[float] = 0.01,
        num_workers: int = 0,
        save_interval: Optional[int] = None,
        save_dir: Optional[str] = None,
        transform: Callable[..., Any] = None,
        collate_fn: Callable[[List[T]], Any] = None,
    ) -> None:
        super().__init__(model, criterion, optimizer, device, batch_size, num_epochs, stop_loss, num_workers, save_interval, save_dir, transform, collate_fn)

    def _fit(self, data_loader) -> float:
        """
        Internal method to fit the model on data for n epochs, with early stopping.

        Parameters
        ----------
        data_loader : DataLoader
            Data loader providing training samples.

        Returns
        -------
        float
            The loss value of the trained model.
        """
        loss_value = 1e9
        for epoch in range(self.num_epochs):
            loss_value = self.train_epoch(data_loader)
            if self.save_interval is not None and (epoch + 1) % self.save_interval == 0:
                if self.save_dir is None:
                    raise ValueError("save_dir should not be None if save_interval is not None.")
                # self.save(epoch + 1)
            if self.stop_loss is not None and loss_value < self.stop_loss:
                break
        return loss_value

    def fit(self, data_loader: DataLoader = None, X: List[Any] = None, y: List[int] = None, confidence: List[float] = None) -> float:
        """
        Train the model.

        Parameters
        ----------
        data_loader : DataLoader, optional
            The data loader used for training, by default None.
        X : List[Any], optional
            The input data, by default None.
        y : List[int], optional
            The target data, by default None.

        Returns
        -------
        float
            The loss value of the trained model.
        """
        if data_loader is None:
            if X is None:
                raise ValueError("data_loader and X can not be None simultaneously.")
            else:
                data_loader = self._ws_data_loader(X, y, confidence)
        return self._fit(data_loader)

    def _ws_data_loader(self, X: List[Any], y: List[int] = None, confidence: List[float] = None, shuffle: bool = True,) -> DataLoader:
        """
        Generate a DataLoader for user-provided input and target data.

        Parameters
        ----------
        X : List[Any]
            Input samples.
        y : List[int], optional
            Target labels. If None, dummy labels are created, by default None.
        Confidence: List[float], optional
            Confidence of labels. If None, dummy labels are created, by default None.

        Returns
        -------
        DataLoader
            A DataLoader providing batches of (X, y, confidence) tuples.
        """

        if X is None:
            raise ValueError("X should not be None.")
        if y is None:
            y = [0] * len(X)
        if confidence is None:
            confidence = [1.0] * len(X)
        if not (len(y) == len(X) == len(confidence)):
            raise ValueError("X and y and confidence should have equal length.")

        dataset = WeaklyClassificationDataset(X, y, confidence, transform=self.transform)
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=int(self.num_workers), collate_fn=self.collate_fn,)
        return data_loader

    def train_epoch(self, data_loader: DataLoader) -> float:
        """
        Train the model for one epoch.

        Parameters
        ----------
        data_loader : DataLoader
            The data loader used for training.

        Returns
        -------
        float
            The loss value of the trained model.
        """
        model = self.model
        criterion = self.criterion
        optimizer = self.optimizer
        device = self.device

        model.train()

        total_loss, total_num = 0.0, 0
        for data, target, confidence in data_loader:
            data, target, confidence = data.to(device), target.to(device), confidence.to(device)
            out = model(data)
            loss = torch.mean(criterion(out, target) * confidence)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * data.size(0)
            total_num += data.size(0)

        return total_loss / total_num
