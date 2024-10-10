from typing import Any, Callable, List, Tuple

from torch.utils.data import Dataset
import torch


class ClassificationDataset(Dataset):

    def __init__(self,
                 X: List[Any],
                 Y: List[int],
                 transform: Callable[..., Any] = None):
        """
        Initialize the dataset used for classification task.

        Parameters
        ----------
        X : List[Any]
            The input data.
        Y : List[int]
            The target data.
        transform : Callable[..., Any], optional
            A function/transform that takes in an object and returns a transformed version. Defaults to None.
        """
        if (not isinstance(X, list)) or (not isinstance(Y, list)):
            raise ValueError("X and Y should be of type list.")
        if len(X) != len(Y):
            raise ValueError("Length of X and Y must be equal.")

        self.X = X
        self.Y = Y
        self.transform = transform

    def __len__(self) -> int:
        """
        Return the length of the dataset.

        Returns
        -------
        int
            The length of the dataset.
        """
        return len(self.X)

    def __getitem__(self, index: int) -> Tuple[Any, torch.Tensor]:
        """
        Get the item at the given index.

        Parameters
        ----------
        index : int
            The index of the item to get.

        Returns
        -------
        Tuple[Any, torch.Tensor]
            A tuple containing the object and its label.
        """
        if index >= len(self):
            raise ValueError("index range error")

        x = self.X[index]
        if self.transform is not None:
            x = self.transform(x)

        y = self.Y[index]

        return x, y


class WeaklyClassificationDataset(Dataset):

    def __init__(self,
                 X: List[Any],
                 Y: List[int],
                 Confidence: List[float],
                 transform: Callable[..., Any] = None):
        """
        Initialize the dataset used for classification task.

        Parameters
        ----------
        X : List[Any]
            The input data.
        Y : List[int]
            The target data.
        Confidence:List[float]
            The confidence of if the y is correct.
        transform : Callable[..., Any], optional
            A function/transform that takes in an object and returns a transformed version. Defaults to None.
        """
        if (not isinstance(X, list)) or (not isinstance(
                Y, list)) or (not isinstance(Confidence, list)):
            raise ValueError("X,Y and Confidence should be of type list.")
        if not (len(X) == len(Y) == len(Confidence)):
            raise ValueError("Length of X,Y and Confidence must be equal.")

        self.X = X
        self.Y = Y
        self.Confidence = Confidence
        self.transform = transform

    def __len__(self) -> int:
        """
        Return the length of the dataset.

        Returns
        -------
        int
            The length of the dataset.
        """
        return len(self.X)

    def __getitem__(self, index: int) -> Tuple[Any, torch.Tensor]:
        """
        Get the item at the given index.

        Parameters
        ----------
        index : int
            The index of the item to get.

        Returns
        -------
        Tuple[Any, torch.Tensor]
            A tuple containing the object and its label.
        """
        if index >= len(self):
            raise ValueError("index range error")

        x = self.X[index]
        if self.transform is not None:
            x = self.transform(x)

        y = self.Y[index]
        confidence = self.Confidence[index]

        return x, y, confidence
