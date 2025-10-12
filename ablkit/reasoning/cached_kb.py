from .kb import KBBase
import os
from pathlib import Path
import pickle
import bisect
from collections import defaultdict
from itertools import combinations, product
from multiprocessing import Pool
from typing import Any, Callable, List, Optional

import numpy as np
from ..utils.cache import abl_cache
from ..utils.logger import print_log
from ..utils.utils import flatten, hamming_dist, reform_list, to_hashable


class CachedKB(KBBase):
    """
    Knowledge base with a ground KB (GKB). Ground KB is a knowledge base prebuilt upon
    class initialization, storing all potential candidates along with their respective
    reasoning result. Ground KB can accelerate abductive reasoning in ``abduce_candidates``.

    Parameters
    ----------
    pseudo_label_list : List[Any]
        Refer to class ``KBBase``.
    GKB_len_list : List[int]
        List of possible lengths for pseudo-labels of an example.
    max_err : float, optional
        Refer to class ``KBBase``.

    Notes
    -----
    Users can also inherit from this class to build their own knowledge base. Similar
    to ``KBBase``, users are only required to provide the ``pseudo_label_list`` and override
    the ``logic_forward`` function. Additionally, users should provide the ``GKB_len_list``.
    After that, other operations (e.g. auto-construction of GKB, and how to perform
    abductive reasoning) will be automatically set up.
    """

    def __init__(
        self,
        pseudo_label_list: List[Any],
        GKB_len_list: List[int],
        max_err: float = 1e-10,
        kb_file_path: Path = None,
    ):
        super().__init__(pseudo_label_list, max_err)
        if not isinstance(GKB_len_list, list):
            raise TypeError("GKB_len_list should be list, but got {type(GKB_len_list)}")
        if self._num_args == 2:
            raise NotImplementedError("GroundKB only supports 1-argument logic_forward, but got " + f"{self._num_args}-argument logic_forward")

        if kb_file_path and os.path.exists(kb_file_path):
            self.load_kb(kb_file_path)
        else:
            self.GKB_len_list = GKB_len_list
            self.GKB = {}
            X, Y = self._get_GKB()
            for x, y in zip(X, Y):
                self.GKB.setdefault(len(x), defaultdict(list))[y].append(x)
            self.save_kb(kb_file_path)

    def _get_XY_list(self, args):
        pre_x, post_x_it = args[0], args[1]
        XY_list = []
        for post_x in post_x_it:
            x = (pre_x,) + post_x
            y = self.logic_forward(x)
            if y is not None:
                XY_list.append((x, y))
        return XY_list

    def _get_GKB(self):
        """
        Prebuild the GKB according to ``pseudo_label_list`` and ``GKB_len_list``.
        """
        X, Y = [], []
        for length in self.GKB_len_list:
            arg_list = []
            for pre_x in self.pseudo_label_list:
                post_x_it = product(self.pseudo_label_list, repeat=length - 1)
                arg_list.append((pre_x, post_x_it))
            with Pool(processes=len(arg_list)) as pool:
                ret_list = pool.map(self._get_XY_list, arg_list)
            for XY_list in ret_list:
                if len(XY_list) == 0:
                    continue
                part_X, part_Y = zip(*XY_list)
                X.extend(part_X)
                Y.extend(part_Y)
        if Y and isinstance(Y[0], (int, float)):
            X, Y = zip(*sorted(zip(X, Y), key=lambda pair: pair[1]))
        return X, Y

    def abduce_candidates(
        self,
        pseudo_label: List[Any],
        y: Any,
        x: List[Any],
        max_revision_num: int,
        require_more_revision: int,
    ) -> List[List[Any]]:
        """
        Perform abductive reasoning by directly retrieving compatible candidates from
        the prebuilt GKB. In this way, the time-consuming exhaustive search can be
        avoided.

        Parameters
        ----------
        pseudo_label : List[Any]
            Pseudo-labels of an example (to be revised by abductive reasoning).
        y : Any
            Ground truth of the reasoning result for the example.
        x : List[Any]
            The example (unused in GroundKB).
        max_revision_num : int
            The upper limit on the number of revised labels for each example.
        require_more_revision : int
            Specifies additional number of revisions permitted beyond the minimum required.

        Returns
        -------
        Tuple[List[List[Any]], List[Any]]
            A tuple of two elements. The first element is a list of candidate revisions,
            i.e. revised pseudo-labels of the example. that are compatible with the knowledge
            base. The second element is a list of reasoning results corresponding to each
            candidate, i.e., the outcome of the ``logic_forward`` function.
        """
        if not self.GKB or len(pseudo_label) not in self.GKB_len_list:
            return [], []

        all_candidates, all_reasoning_results = self._find_candidate_GKB(pseudo_label, y)
        if len(all_candidates) == 0:
            return [], []

        cost_list = hamming_dist(pseudo_label, all_candidates)
        min_revision_num = np.min(cost_list)
        revision_num = min(max_revision_num, min_revision_num + require_more_revision)
        idxs = np.where(cost_list <= revision_num)[0]
        candidates = [all_candidates[idx] for idx in idxs]
        reasoning_results = [all_reasoning_results[idx] for idx in idxs]
        return candidates, reasoning_results

    def _find_candidate_GKB(self, pseudo_label: List[Any], y: Any) -> List[List[Any]]:
        """
        Retrieve compatible candidates from the prebuilt GKB. For numerical reasoning results,
        return all candidates and their corresponding reasoning results which fall within the
        [y - max_err, y + max_err] range.
        """
        if isinstance(y, (int, float)):
            potential_candidates = self.GKB[len(pseudo_label)]
            key_list = list(potential_candidates.keys())

            low_key = bisect.bisect_left(key_list, y - self.max_err)
            high_key = bisect.bisect_right(key_list, y + self.max_err)

            all_candidates, all_reasoning_results = [], []
            for key in key_list[low_key:high_key]:
                for candidate in potential_candidates[key]:
                    all_candidates.append(candidate)
                    all_reasoning_results.append(key)
        else:
            all_candidates = self.GKB[len(pseudo_label)][y]
            all_reasoning_results = [y] * len(all_candidates)
        return all_candidates, all_reasoning_results

    def __repr__(self):
        GKB_info_parts = []
        for i in self.GKB_len_list:
            num_candidates = len(self.GKB[i]) if i in self.GKB else 0
            GKB_info_parts.append(f"{num_candidates} candidates of length {i}")
        GKB_info = ", ".join(GKB_info_parts)

        return (
            f"{self.__class__.__name__} is a KB with "
            f"pseudo_label_list={self.pseudo_label_list!r}, "
            f"max_err={self.max_err!r}, "
            f"use_cache={self.use_cache!r}. "
            f"It has a prebuilt GKB with "
            f"GKB_len_list={self.GKB_len_list!r}, "
            f"and there are "
            f"{GKB_info}"
            f" in the GKB."
        )

    def save_kb(self, file_path):
        """
        Save the knowledge base to a file.

        Args:
        file_path (str): The path to the file where the knowledge base will be saved.
        """
        if file_path is None:
            file_path = "."

        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump(self.GKB, f)

    def load_kb(self, file_path):
        """
        Load the knowledge base from a file.

        Args:
        file_path (str): The path to the file from which the knowledge base will be loaded.
        """
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                self.GKB = pickle.load(f)
        else:
            print(f"File {file_path} not found. Starting with an empty knowledge base.")
