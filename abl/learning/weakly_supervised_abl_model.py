# coding: utf-8
from functools import reduce
from typing import Any, List, Optional, Tuple

from scipy.special import softmax
import numpy as np
from functools import reduce
import heapq

from ..utils import flatten
from .abl_model import ABLModel
import time
import wandb
import torch
import torch.nn.functional as F

class WeaklySupervisedABLModel(ABLModel):
    """
    Serialize data and provide a unified interface for different machine learning models.

    Implemented with a wekaly supervised perspective.

    Parameters
    ----------
    base_model : Machine Learning Model
        The base model to use for training and prediction.

    Attributes
    ----------
    classifier_list : List[Any]
        A list of classifiers.

    Methods
    -------
    predict(X: List[List[Any]], mapping: Optional[dict] = None) -> dict
        Predict the labels and probabilities for the given data.
    valid(X: List[List[Any]], Y: List[Any]) -> float
        Calculate the accuracy score for the given data.
    train(X: List[List[Any]], Y: List[Any]) -> float
        Train the model on the given data.
    save(*args, **kwargs) -> None
        Save the model to a file.
    load(*args, **kwargs) -> None
        Load the model from a file.
    """

    def __init__(self, base_model, topK: int = -1, temp: float = 1.0) -> None:
        super().__init__(base_model)
        self.topK = topK
        self.temp = temp

    def train(self, X: List[List[Any]], candidate_set: List[Any], Z: List[Any]) -> float:
        model = self.classifier_list[0]
        topK = self.topK
        loss = 0.0
        Xs, Ys, Confidences = [], [], []
        confidence = 0.0  # The correct candidate's confidence (AVG)
        abduce_acc = 0
        for xs, candidate_set, zs in zip(X, candidate_set, Z):
            # xs: [x1,x2,...,xm] (e.g., digits of an equation)
            # candidate_set: [[y1,y2,...,ym], [y1,y2,...,ym]] (candiates set of possible labels)
            # zs: [y1,y2,...,ym] (ground truth, i.e., label, of xs)
            probs = model.predict_proba(X=xs)
            candidate_probs = self.candidate_confidence(probs, candidate_set)
            candidate_set, candidate_probs = self._topk(
                candidate_set, candidate_probs, topK
            )
            aggregated_label = self.aggregate_(candidate_set, candidate_probs)
            Xs.extend(xs)
            Ys.extend(aggregated_label)
        loss += model.fit(X=Xs, y=Ys)
        confidence /= len(X)
        abduce_acc /= len(X)
        return loss, confidence, abduce_acc
    
    def aggregate_(self, candidate_set: List[List[int]], candidate_probs: List[float]):
        with torch.no_grad():
            label_num = len(candidate_set[0]) 
            aggregate_label = torch.zeros(size=(label_num, 10))
            for candidate, prob in zip(candidate_set, candidate_probs):
                for i, item in enumerate(candidate):
                    aggregate_label[i][item] += prob                   
                    
        return aggregate_label

    def aggregate(self, candidate_set: List[List[int]], candidate_probs: List[float]):
        with torch.no_grad():
            # Convert candidate_set to a tensor for easier manipulation
            candidate_set_tensor = torch.tensor(candidate_set, dtype=torch.long)

            # Find the dimension for one-hot encoding
            vocab_size = 10 ## JUST FOR SIMPLICITY, TO BE OPTIMIZED

            # Convert candidate_probs to a tensor and reshape for broadcasting
            candidate_probs_tensor = torch.tensor(candidate_probs, dtype=torch.float).view(-1, 1, 1)

            # Perform one-hot encoding for the entire candidate_set_tensor at once
            # The shape of one_hot_tensor will be [len(candidate_set), 2, vocab_size]
            one_hot_tensor = torch.nn.functional.one_hot(candidate_set_tensor, num_classes=vocab_size).float()

            # Weight the one-hot encoded tensor by candidate_probs
            weighted_one_hot = one_hot_tensor * candidate_probs_tensor

            # Sum across the batch (candidate sets) to get the soft labels
            soft_labels = torch.sum(weighted_one_hot, dim=0)
            normalized_soft_labels = F.softmax(soft_labels, dim=-1)

        return normalized_soft_labels


    def candidate_confidence(self, probs: List, candidate_set: List) -> List:
        # prob: (equation size, classes num)
        def f(x, prob): return prob[x]
        candidate_probs = [
            reduce(lambda x, y: x + y, map(f, candidate, probs)) / self.temp
            for candidate in candidate_set
        ]
        # candidate_probs = [x / sum(candidate_probs) for x in candidate_probs]  # Normalize
        candidate_probs = softmax(candidate_probs)
        return candidate_probs

    def _topk(self, candidate_set: List[Any], candidate_probs: List[float], K: int = -1) -> Tuple[List[Any], List[float]]:
        """
        Performs a top-k selection from the candidate_set based on candidate_probs. 
        If `K` is set to -1, all candidates are chosen. 
        Returns a tuple containing the selected candidates and their corresponding probabilities.
        """
        if K == -1 or len(candidate_set) <= K:
            topk_candidates, topk_probs = zip(
                *sorted(zip(candidate_set, candidate_probs), key=lambda x: x[1], reverse=True))
            return list(topk_candidates), list(topk_probs)

        # Iterate over all candidates and maintain a heap of size K with the largest probabilities
        heap = []
        for i, (candidate, prob) in enumerate(zip(candidate_set, candidate_probs)):
            if i < K:
                heapq.heappush(heap, (prob, candidate))
            else:
                if prob > heap[0][0]:
                    heapq.heappop(heap)
                    heapq.heappush(heap, (prob, candidate))

        # Extract top-k elements from the heap and reverse them to get the highest probabilities first
        topk_probs, topk_candidates = zip(*sorted(heap, reverse=True))
        return list(topk_candidates), list(topk_probs)
