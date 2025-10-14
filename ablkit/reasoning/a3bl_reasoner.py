"""
This module contains the class A3blReasoner, which is used for minimizing the inconsistency
between the knowledge base and learning models.

Copyright (c) 2025 LAMDA.  All rights reserved.
"""

from typing import Any, Callable, List, Optional, Union, Tuple

import numpy as np
from scipy.special import softmax
import heapq
import torch
import torch.nn.functional as F
import pdb
from ..data.structures import ListData
from .reasoner import Reasoner


def confidence_dist(pred_probs: np.ndarray, candidate_idxs: List[List[Any]], temp: float = 1.0) -> np.ndarray:
    candidates_array = np.array(candidate_idxs)
    _, symbol_num = candidates_array.shape
    row_indices = np.arange(symbol_num)[:, np.newaxis]
    selected_probs = pred_probs[row_indices, candidates_array.T]
    candidate_probs = np.sum(selected_probs, axis=0) / temp
    return softmax(candidate_probs)


def confidence_dist_multi_label(pred_probs: np.ndarray, candidate_idxs: List[List[Any]], temp: float = 1.0) -> np.ndarray:
    candidate_probs = pred_probs @ np.array(candidate_idxs).T / temp
    return softmax(candidate_probs.squeeze(axis=0))



class A3BLReasoner(Reasoner):  # TODO
    """
    Reasoner for minimizing the inconsistency between the knowledge base and learning models.

    Parameters
    ----------
    kb : class KBBase
        The knowledge base to be used for reasoning.
    dist_func : Union[str, Callable], optional
        The distance function used to determine the cost list between each
        candidate and the given prediction. The cost is also referred to as a consistency
        measure, wherein the candidate with lowest cost is selected as the final
        abduced label. It can be either a string representing a predefined distance
        function or a callable function. The available predefined distance functions:
        'hamming' | 'confidence' | 'avg_confidence'. 'hamming' directly calculates the
        Hamming distance between the predicted pseudo-label in the data example and each
        candidate. 'confidence' and 'avg_confidence' calculates the confidence distance
        between the predicted probabilities in the data example and each candidate, where
        the confidence distance is defined as 1 - the product of prediction probabilities
        in 'confidence' and 1 - the average of prediction probabilities in 'avg_confidence'.
        Alternatively, the callable function should have the signature
        ``dist_func(data_example, candidates, candidate_idxs, reasoning_results)`` and must
        return a cost list. Each element in this cost list should be a numerical value
        representing the cost for each candidate, and the list should have the same length
        as candidates. Defaults to 'confidence'.
    idx_to_label : dict, optional
        A mapping from index in the base model to label. If not provided, a default
        order-based index to label mapping is created. Defaults to None.
    max_revision : Union[int, float], optional
        The upper limit on the number of revisions for each data example when
        performing abductive reasoning. If float, denotes the fraction of the total
        length that can be revised. A value of -1 implies no restriction on the
        number of revisions. Defaults to -1.
    require_more_revision : int, optional
        Specifies additional number of revisions permitted beyond the minimum required
        when performing abductive reasoning. Defaults to 0.
    use_zoopt : bool, optional
        Whether to use ZOOpt library during abductive reasoning. Defaults to False.
    """

    def __init__(
        self,
        kb,
        dist_func="confidence",
        idx_to_label=None,
        max_revision: Union[int, float] = -1,
        require_more_revision: int = 0,
        use_zoopt: bool = False,
        topK: int = 16,
        temperature: float = 0.2,
        multi_label: bool = False,
    ):
        super().__init__(kb, dist_func, idx_to_label, max_revision, require_more_revision, use_zoopt)
        self.topK = topK
        self.temperature = temperature
        self.class_num = len(self.kb.pseudo_label_list)
        self.multi_label = multi_label
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _candidates_idxs(self, candidates: List[List[Any]]):
        return [[self.label_to_idx[x] for x in c] for c in candidates]

    def _topk(self, candidates: List[Any], candidate_probs: np.ndarray, K: int = -1) -> Tuple[List[List[Any]], List[Any]]:
        """
        Performs a top-k selection from the candidate_set based on candidate_probs.
        If `K` is set to -1, all candidates are chosen.
        Returns a tuple containing the selected candidates and their corresponding probabilities.
        """

        if K == -1 or len(candidates) <= K:
            return candidates, candidate_probs

        # Iterate over all candidates and maintain a heap of size K with the largest probabilities
        heap = []
        for i, (candidate, prob) in enumerate(zip(candidates, candidate_probs)):
            if i < K:
                heapq.heappush(heap, (prob, candidate))
            else:
                if prob > heap[0][0]:
                    heapq.heappop(heap)
                    heapq.heappush(heap, (prob, candidate))

        # Extract top-k elements from the heap and reverse them to get the highest probabilities first
        topk_probs, topk_candidates = zip(*heap)
        return list(topk_candidates), list(topk_probs)

    def multi_label_aggregate(self, candidates: List[List[int]], candidate_probs: List[float]):
        """
        An multi-label version of A3BL.
        """
        with torch.no_grad():
            symbol_num = len(candidates[0])
            aggregate_label = torch.zeros(size=(symbol_num, 1))
            for candidate, prob in zip(candidates, candidate_probs):
                for i, item in enumerate(candidate):
                    if item == 1:
                        aggregate_label[i] += prob
        return list(aggregate_label.unbind(1))

    def aggregate(self, candidates: List[List[int]], candidate_probs: List[float]):
        with torch.no_grad():
            candidates_tensor = torch.tensor(candidates, device=self.device, dtype=torch.long)
            probs_tensor = torch.tensor(candidate_probs, device=self.device, dtype=torch.float32)
            one_hot = F.one_hot(candidates_tensor, num_classes=self.class_num).float()  # [N, M, C]
            weighted_one_hot = one_hot * probs_tensor.unsqueeze(-1).unsqueeze(-1)  # [N, M, C]
            aggregate_label = weighted_one_hot.sum(dim=0)  # [M, C]
        return [tensor.cpu() for tensor in aggregate_label.unbind(0)]

    def abduce(self, data_example: ListData) -> List[Any]:
        """
        Perform abduction and get soft label distribution (given by all valid candidates that satisfy the underlying rules).

        Parameters
        ----------
        data_example : ListData
            Data example.

        Returns
        -------

        List[Any]
            A revised soft label which is aggregated from valid candidates.

        List[Any]
            A revised pseudo-labels of the example through abductive reasoning, which is compatible
            with the knowledge base.


        """
        max_revision_num = data_example.elements_num("pred_pseudo_label")
        max_revision_num = self._get_max_revision_num(self.max_revision, max_revision_num)
        candidates, _ = self.kb.abduce_candidates(
            pseudo_label=data_example.pred_pseudo_label,
            y=data_example.Y,
            x=data_example.X,
            max_revision_num=max_revision_num,
            require_more_revision=self.require_more_revision,
        )

        if len(candidates) == 0:
            return [], []
        
        confidence_dist_cal = confidence_dist if not self.multi_label else confidence_dist_multi_label

        candidate_probs = confidence_dist_cal(data_example.pred_prob, self._candidates_idxs(candidates), self.temperature)
        topk_candidates, topk_candidates_probs = self._topk(candidates, candidate_probs, self.topK)
        aggregated_labels = (
            self.aggregate(topk_candidates, topk_candidates_probs)
            if not self.multi_label
            else self.multi_label_aggregate(topk_candidates, topk_candidates_probs)
        )
        return aggregated_labels, topk_candidates[0]

    def batch_abduce(self, data_examples: ListData) -> List[List[Any]]:
        """
        Perform abductive reasoning on the given prediction data examples.
        For detailed information, refer to ``abduce``.
        """
        abduced_soft_label, abduced_pseudo_label = zip(*[self.abduce(data_example) for data_example in data_examples])
        data_examples.abduced_soft_label = abduced_soft_label
        data_examples.abduced_pseudo_label = abduced_pseudo_label
        return abduced_soft_label

    def __call__(self, data_examples: ListData) -> List[List[Any]]:
        return self.batch_abduce(data_examples)
