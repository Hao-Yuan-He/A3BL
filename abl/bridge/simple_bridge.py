from abl.evaluation import BaseMetric
from abl.learning import ABLModel
from abl.reasoning import ReasonerBase, WeaklySupervisedReasoner
from ..learning import ABLModel, WeaklySupervisedABLModel
from ..reasoning import ReasonerBase
from ..evaluation import BaseMetric
from .base_bridge import BaseBridge
from typing import List, Union, Any, Tuple, Dict, Optional
from numpy import ndarray
import wandb
from torch.utils.data import DataLoader
from ..dataset import BridgeDataset
from ..utils.logger import print_log
import numpy as np


class SimpleBridge(BaseBridge):
    def __init__(
        self,
        model: ABLModel,
        abducer: ReasonerBase,
        metric_list: List[BaseMetric],
    ) -> None:
        super().__init__(model, abducer)
        self.metric_list = metric_list

    def predict(self, X) -> Tuple[List[List[Any]], ndarray]:
        pred_res = self.model.predict(X)
        pred_idx, pred_prob = pred_res["label"], pred_res["prob"]
        return pred_idx, pred_prob

    def abduce_pseudo_label(
        self,
        pred_prob: ndarray,
        pred_pseudo_label: List[List[Any]],
        Y: List[Any],
        max_revision: int = -1,
        require_more_revision: int = 0,
    ) -> List[List[Any]]:
        return self.abducer.batch_abduce(pred_prob, pred_pseudo_label, Y, max_revision, require_more_revision)

    def idx_to_pseudo_label(self, idx: List[List[Any]], mapping: Dict = None) -> List[List[Any]]:
        if mapping is None:
            mapping = self.abducer.mapping
        return [[mapping[_idx] for _idx in sub_list] for sub_list in idx]

    def pseudo_label_to_idx(self, pseudo_label: List[List[Any]], mapping: Dict = None) -> List[List[Any]]:
        if mapping is None:
            mapping = self.abducer.remapping

        def recursive_map(func, nested_list):
            if isinstance(nested_list, (list, tuple)):
                return [recursive_map(func, x) for x in nested_list]
            else:
                return func(nested_list)

        return recursive_map(lambda x: mapping[x], pseudo_label)

    def train(
        self,
        train_data: Tuple[List[List[Any]], Optional[List[List[Any]]], List[List[Any]]],
        epochs: int = 50,
        batch_size: Union[int, float] = -1,
        eval_interval: int = 1,
        test_data: Tuple[List[List[Any]], Optional[List[List[Any]]], List[List[Any]]] = None,
    ):
        dataset = BridgeDataset(*train_data)
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=lambda data_list: [list(data) for data in zip(*data_list)],
        )

        for epoch in range(epochs):
            for seg_idx, (X, Z, Y) in enumerate(data_loader):
                pred_idx, pred_prob = self.predict(X)
                pred_pseudo_label = self.idx_to_pseudo_label(pred_idx)
                abduced_pseudo_label = self.abduce_pseudo_label(pred_prob, pred_pseudo_label, Y)
                abduced_label = self.pseudo_label_to_idx(abduced_pseudo_label)
                loss = self.model.train(X, abduced_label)
                abduce_acc = np.mean([i==j for i,j in zip(abduced_label, self.pseudo_label_to_idx(Z))])
                print_log(
                    f"Epoch(train) [{epoch + 1}] [{(seg_idx + 1):3}/{len(data_loader)}] model loss is {loss:.5f}",
                    logger="current",
                )
                wandb.log({"train loss": loss, "abduce_acc":abduce_acc})

            if (epoch + 1) % eval_interval == 0 or epoch == epochs - 1:
                print_log(f"Evaluation start: Epoch(val) [{epoch}]", logger="current")
                self.valid(train_data)
                if test_data:
                    self.test(test_data)

    def _valid(self, data_loader, tag=""):
        for X, Z, Y in data_loader:
            pred_idx, pred_prob = self.predict(X)
            pred_pseudo_label = self.idx_to_pseudo_label(pred_idx)
            data_samples = dict(
                pred_idx=pred_idx,
                pred_prob=pred_prob,
                pred_pseudo_label=pred_pseudo_label,
                gt_pseudo_label=Z,
                Y=Y,
                logic_forward=self.abducer.kb.logic_forward,
            )
            for metric in self.metric_list:
                metric.process(data_samples)

        res = dict()
        for metric in self.metric_list:
            res.update(metric.evaluate())
        wandb.log({f"{k}/{tag}": v for k, v in res.items()})
        msg = "Evaluation ended, "
        try:
            for k, v in res.items():
                msg += k + f": {v:.3f} "
            print_log(msg, logger="current")
        except:
            pass

    def valid(self, valid_data, batch_size=1000, tag="valid"):
        dataset = BridgeDataset(*valid_data)
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=lambda data_list: [list(data) for data in zip(*data_list)],
        )
        self._valid(data_loader, tag)

    def test(self, test_data, batch_size=1000):
        self.valid(test_data, batch_size, tag="test")


class WeaklySupervisedBridge(SimpleBridge):
    def __init__(self, model: WeaklySupervisedABLModel, abducer: WeaklySupervisedReasoner, metric_list: List[BaseMetric]) -> None:
        assert isinstance(abducer, WeaklySupervisedReasoner), f"abducer should be an instance of WeaklySupervisedReasoner but get {type(abducer)}"
        super().__init__(model, abducer, metric_list)

    def abduce_candidates_set(self, pred_prob: ndarray, pred_pseudo_label: List[List[Any]], Y: List[Any], max_revision: int = -1, require_more_revision: int = 1) -> List[List[Any]]:
        return self.abducer.batch_abduce_candidates_set(pred_prob, pred_pseudo_label, Y, max_revision, require_more_revision)

    def train(
        self,
        train_data: Tuple[List[List[Any]], Optional[List[List[Any]]], List[List[Any]]],
        epochs: int = 50,
        batch_size: Union[int, float] = -1,
        eval_interval: int = 1,
        more_revision: int = 3,
        test_data=None,
    ):
        dataset = BridgeDataset(*train_data)
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=lambda data_list: [list(data) for data in zip(*data_list)],
        )

        for epoch in range(epochs):
            for seg_idx, (X, Z, Y) in enumerate(data_loader):
                pred_idx, pred_prob = self.predict(X)
                pred_pseudo_label = self.idx_to_pseudo_label(pred_idx)
                abduced_candidates_set = self.abduce_candidates_set(pred_prob, pred_pseudo_label, Y, require_more_revision=more_revision)
                abduced_candidates_set = self.pseudo_label_to_idx(abduced_candidates_set)
                loss,confidence, abduce_acc = self.model.train(X, abduced_candidates_set, Z=self.pseudo_label_to_idx(Z))

                print_log(
                    f"Epoch(train) [{epoch + 1}] [{(seg_idx + 1):3}/{len(data_loader)}] model loss is {loss:.5f}",
                    logger="current",
                )
                candidate_set_size = sum([len(x) for x in abduced_candidates_set]) / len(abduced_candidates_set)
                wandb.log({"train loss": loss, "candidate set size": candidate_set_size, "Confidence": confidence, "abduce_acc":abduce_acc})

            if (epoch + 1) % eval_interval == 0 or epoch == epochs - 1:
                print_log(f"Evaluation start: Epoch(val) [{epoch}]", logger="current")
                self.valid(train_data)
                if test_data:
                    self.test(test_data)
