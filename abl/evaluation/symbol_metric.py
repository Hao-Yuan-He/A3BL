from typing import Optional, Sequence, Callable
from .base_metric import BaseMetric
import numpy as np
import wandb
from itertools import chain 

class SymbolMetric(BaseMetric):
    def __init__(self, prefix: Optional[str] = None) -> None:
        super().__init__(prefix)
        self.y_pred = []
        self.y_gt = []
    def process(self, data_samples: Sequence[dict]) -> None:
        pred_pseudo_label = data_samples["pred_pseudo_label"]

        gt_pseudo_label = data_samples["gt_pseudo_label"]

        if not len(pred_pseudo_label) == len(gt_pseudo_label):
            raise ValueError("lengthes of pred_pseudo_label and gt_pseudo_label should be equal")
        self.y_gt.extend(gt_pseudo_label)
        self.y_pred.extend(pred_pseudo_label)
        for pred_z, z in zip(pred_pseudo_label, gt_pseudo_label):
            correct_num = 0
            for pred_symbol, symbol in zip(pred_z, z):
                if pred_symbol == symbol:
                    correct_num += 1
            self.results.append(correct_num / len(z))
        
    
    def compute_metrics(self, results: list) -> dict:
        metrics = dict()
        metrics["character_accuracy"] = sum(results) / len(results)
        cm = wandb.plot.confusion_matrix(preds=list(chain(*self.y_pred)), y_true=list(chain(*self.y_gt)))
        metrics['Confusing Matrix'] = cm
        return metrics