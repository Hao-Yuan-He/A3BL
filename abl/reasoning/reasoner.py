import numpy as np
from typing import Dict, List
from zoopt import Dimension, Objective, Parameter, Opt
from .kb import KBBase
from ..utils.utils import (
    confidence_dist,
    flatten,
    reform_idx,
    hamming_dist,
    calculate_revision_num,
)


class ReasonerBase:
    def __init__(self, kb, dist_func="hamming", mapping=None, use_zoopt=False):
        """
        Root class for all reasoner in the ABL system.

        Parameters
        ----------
        kb : KBBase
            The knowledge base to be used for reasoning.
        dist_func : str, optional
            The distance function to be used. Can be "hamming" or "confidence". Default is "hamming".
        mapping : dict, optional
            A mapping of indices to labels. If None, a default mapping is generated.
        use_zoopt : bool, optional
            Whether to use the Zoopt library for optimization. Default is False.

        Raises
        ------
        NotImplementedError
            If the specified distance function is neither "hamming" nor "confidence".
        """

        if not (dist_func == "hamming" or dist_func == "confidence"):
            raise NotImplementedError  # Only hamming or confidence distance is available.

        self.kb = kb
        self.dist_func = dist_func
        self.use_zoopt = use_zoopt
        if mapping is None:
            self.mapping = {index: label for index, label in enumerate(self.kb.pseudo_label_list)}
        else:
            self.mapping = mapping
        self.remapping = dict(zip(self.mapping.values(), self.mapping.keys()))

    def _get_cost_list(self, pred_pseudo_label, pred_prob, candidates):
        """
        Get the list of costs between pseudo label and each candidate.

        Parameters
        ----------
        pred_pseudo_label : list
            The pseudo label to be used for computing costs of candidates.
        pred_prob : list
            Probabilities of the predictions. Used when distance function is "confidence".
        candidates : list
            List of candidate abduction result.

        Returns
        -------
        numpy.ndarray
            Array of computed costs for each candidate.
        """
        if self.dist_func == "hamming":
            return hamming_dist(pred_pseudo_label, candidates)

        elif self.dist_func == "confidence":
            candidates = [[self.remapping[x] for x in c] for c in candidates]
            return confidence_dist(pred_prob, candidates)

    def _get_one_candidate(self, pred_pseudo_label, pred_prob, candidates):
        """
        Get one candidate. If multiple candidates exist, return the one with minimum cost.

        Parameters
        ----------
        pred_pseudo_label : list
            The pseudo label to be used for selecting a candidate.
        pred_prob : list
            Probabilities of the predictions.
        candidates : list
            List of candidate abduction result.

        Returns
        -------
        list
            The chosen candidate based on minimum cost.
            If no candidates, an empty list is returned.
        """
        if len(candidates) == 0:
            return []
        elif len(candidates) == 1:
            return candidates[0]
        else:
            cost_array = self._get_cost_list(pred_pseudo_label, pred_prob, candidates)
            candidate = candidates[np.argmin(cost_array)]
            return candidate

    def zoopt_revision_score(self, symbol_num, pred_pseudo_label, pred_prob, y, sol):
        """
        Get the revision score for a single solution.

        Parameters
        ----------
        symbol_num : int
            Number of total symbols.
        pred_pseudo_label : list
            List of predicted pseudo labels.
        pred_prob : list
            List of probabilities for predicted results.
        y : any
            Ground truth for the predicted results.
        sol : array-like
            Solution to evaluate.

        Returns
        -------
        float
            The revision score for the given solution.
        """
        revision_idx = np.where(sol.get_x() != 0)[0]
        candidates = self.revise_by_idx(pred_pseudo_label, y, revision_idx)
        if len(candidates) > 0:
            return np.min(self._get_cost_list(pred_pseudo_label, pred_prob, candidates))
        else:
            return symbol_num

    def _constrain_revision_num(self, solution, max_revision_num):
        x = solution.get_x()
        return max_revision_num - x.sum()

    def zoopt_get_solution(self, symbol_num, pred_pseudo_label, pred_prob, y, max_revision_num):
        """Get the optimal solution using the Zoopt library.

        Parameters
        ----------
        symbol_num : int
            Number of total symbols.
        pred_pseudo_label : list
            List of predicted pseudo labels.
        pred_prob : list
            List of probabilities for predicted results.
        y : any
            Ground truth for the predicted results.
        max_revision_num : int
            Maximum number of revisions to use.

        Returns
        -------
        array-like
            The optimal solution, i.e., where to revise predict pseudo label.
        """
        print(symbol_num)
        dimension = Dimension(size=symbol_num, regs=[[0, 1]] * symbol_num, tys=[False] * symbol_num)
        objective = Objective(
            lambda sol: self.zoopt_revision_score(symbol_num, pred_pseudo_label, pred_prob, y, sol), dim=dimension, constraint=lambda sol: self._constrain_revision_num(sol, max_revision_num),
        )
        parameter = Parameter(budget=100, intermediate_result=False, autoset=True)
        solution = Opt.min(objective, parameter).get_x()
        return solution

    def revise_by_idx(self, pred_pseudo_label, y, revision_idx):
        """
        Revise the pseudo label according to the given indices.

        Parameters
        ----------
        pred_pseudo_label : list
            List of predicted pseudo labels.
        y : any
            Ground truth for the predicted results.
        revision_idx : array-like
            Indices of the revisions to retrieve.

        Returns
        -------
        list
            The revisions according to the given indices.
        """
        return self.kb.revise_by_idx(pred_pseudo_label, y, revision_idx)

    def abduce(self, pred_prob, pred_pseudo_label, y, max_revision=-1, require_more_revision=0):
        """
        Perform revision by abduction on the given data.

        Parameters
        ----------
        pred_prob : list
            List of probabilities for predicted results.
        pred_pseudo_label : list
            List of predicted pseudo labels.
        y : any
            Ground truth for the predicted results.
        max_revision : int or float, optional
            Maximum number of revisions to use. If float, represents the fraction of total revisions to use.
            If -1, any revisions are allowed. Defaults to -1.
        require_more_revision : int, optional
            Number of additional revisions to require. Defaults to 0.

        Returns
        -------
        list
            The abduced revisions.
        """
        symbol_num = len(flatten(pred_pseudo_label))
        max_revision_num = calculate_revision_num(max_revision, symbol_num)

        if self.use_zoopt:
            solution = self.zoopt_get_solution(symbol_num, pred_pseudo_label, pred_prob, y, max_revision_num)
            revision_idx = np.where(solution != 0)[0]
            candidates = self.revise_by_idx(pred_pseudo_label, y, revision_idx)
        else:
            candidates = self.kb.abduce_candidates(pred_pseudo_label, y, max_revision_num, require_more_revision)

        candidate = self._get_one_candidate(pred_pseudo_label, pred_prob, candidates)
        return candidate

    def batch_abduce(self, pred_prob, pred_pseudo_label, Y, max_revision=-1, require_more_revision=0):
        """
        Perform abduction on the given data in batches.

        Parameters
        ----------
        pred_prob : list
            List of probabilities for predicted results.
        pred_pseudo_label : list
            List of predicted pseudo labels.
        Y : list
            List of ground truths for the predicted results.
        max_revision : int or float, optional
            Maximum number of revisions to use. If float, represents the fraction of total revisions to use.
            If -1, use all revisions. Defaults to -1.
        require_more_revision : int, optional
            Number of additional revisions to require. Defaults to 0.

        Returns
        -------
        list
            The abduced revisions in batches.
        """
        return [self.abduce(_pred_prob, _pred_pseudo_label, _Y, max_revision, require_more_revision) for _pred_prob, _pred_pseudo_label, _Y in zip(pred_prob, pred_pseudo_label, Y)]

    def __call__(self, pred_prob, pred_pseudo_label, Y, max_revision=-1, require_more_revision=0):
        return self.batch_abduce(pred_prob, pred_pseudo_label, Y, max_revision, require_more_revision)

    def set_remapping(self):
        self.remapping = dict(zip(self.mapping.values(), self.mapping.keys()))

class WeaklySupervisedReasoner(ReasonerBase):
    def __init__(self, kb: KBBase, dist_func: str = "hamming", mapping: Dict = None, use_zoopt: bool = False):
        super().__init__(kb, dist_func, mapping, use_zoopt)

    def abduce_candidates_set(self, pred_prob: List, pred_pseudo_label: List, y, max_revision=-1, require_more_revision=0) -> List:
        """
        Perform revision by abduction on the given data.

        Parameters
        ----------
        pred_prob : list
            List of probabilities for predicted results.
        pred_pseudo_label : list
            List of predicted pseudo labels.
        y : any
            Ground truth for the predicted results.
        max_revision : int or float, optional
            Maximum number of revisions to use. If float, represents the fraction of total revisions to use.
            If -1, any revisions are allowed. Defaults to -1.
        require_more_revision : int, optional
            Number of additional revisions to require. Defaults to 0.

        Returns
        -------
        list
            The abduced revisions.
        """
        symbol_num = len(flatten(pred_pseudo_label))
        max_revision_num = calculate_revision_num(max_revision, symbol_num)

        if self.use_zoopt:
            solution = self.zoopt_get_solution(symbol_num, pred_pseudo_label, pred_prob, y, max_revision_num)
            revision_idx = np.where(solution != 0)[0]
            candidates = self.revise_by_idx(pred_pseudo_label, y, revision_idx)
        else:
            candidates = self.kb.abduce_candidates(pred_pseudo_label, y, max_revision_num, require_more_revision)
        return candidates

    def batch_abduce_candidates_set(self, pred_prob: List, pred_pseudo_label: List, Y: List, max_revision=-1, require_more_revision=0) -> List:
        """
        Perform abduction on the given data in batches.

        Parameters
        ----------
        pred_prob : list
            List of probabilities for predicted results.
        pred_pseudo_label : list
            List of predicted pseudo labels.
        Y : list
            List of ground truths for the predicted results.
        max_revision : int or float, optional
            Maximum number of revisions to use. If float, represents the fraction of total revisions to use.
            If -1, use all revisions. Defaults to -1.
        require_more_revision : int, optional
            Number of additional revisions to require. Defaults to 0.

        Returns
        -------
        list
            The abduced revisions in batches.
        """
        return [self.abduce_candidates_set(_pred_prob, _pred_pseudo_label, _Y, max_revision, require_more_revision) for _pred_prob, _pred_pseudo_label, _Y in zip(pred_prob, pred_pseudo_label, Y)]
