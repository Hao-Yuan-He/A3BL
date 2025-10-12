"""
This module contains the class ABLModel, which provides a unified interface for different
machine learning models.

Copyright (c) 2025 LAMDA.  All rights reserved.
"""

from .abl_model import ABLModel
from ..data.structures import ListData
from itertools import chain


class A3BLModel(ABLModel):
    """
    Serialize data and provide a unified interface for different machine learning models.

    Parameters
    ----------
    base_model : Machine Learning Model
        The machine learning base model used for training and prediction. This model should
        implement the ``fit`` and ``predict`` methods. It's recommended, but not required, for the
        model to also implement the ``predict_proba`` method for generating
        predictions on the probabilities.
    """

    def __init__(self, base_model):
        super().__init__(base_model)


    def train(self, data_examples: ListData) -> float:
        """
        Train the model on the given data.

        Parameters
        ----------
        data_examples : ListData
            A batch of data to train on, which typically contains the data, ``X``, and the
            corresponding soft labels, ``abduced_soft_label``.

        Returns
        -------
        float
            The loss value of the trained model.
        """
        data_X = data_examples.flatten("X")
        data_y = list(chain(*data_examples.abduced_soft_label))
        return self.base_model.fit(X=data_X, y=data_y)
