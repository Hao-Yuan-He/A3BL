from ablkit.bridge import A3BLBridge
from ablkit.data import ListData
from ablkit.utils import print_log

import wandb

from datasets import MixedDataset

class MixedBridge(A3BLBridge):
    def __init__(self, model, reasoner, metric_list):
        super().__init__(model, reasoner, metric_list) 
        
        
    def data_preprocess(self, prefix, data):
        if isinstance(data, ListData):
            data_examples = data
            if not (
                hasattr(data_examples, "X")
                and hasattr(data_examples, "gt_pseudo_label")
                and hasattr(data_examples, "Y")
                and hasattr(data_examples, "Op")
            ):
                raise ValueError(
                    f"{prefix}data should have X, gt_pseudo_label, Y and Op attribute but "
                    f"only {data_examples.all_keys()} are provided."
                )
        else:
            X, gt_pseudo_label, Y, Op = data
            data_examples = ListData(X=X, gt_pseudo_label=gt_pseudo_label, Y=Y, Op=Op)

        return data_examples
    
    def idx_to_pseudo_label(self, data_examples):
        pred_idx, ops = data_examples.pred_idx, data_examples.Op
        data_examples.pred_pseudo_label = [
            [self.reasoner.idx_to_label[_idx] for _idx in sub_list] + [op] for sub_list, op in zip(pred_idx, ops)
        ]
        return data_examples.pred_pseudo_label
    
    def _valid_idx_to_pseudo_label(self, data_examples):
        pred_idx = data_examples.pred_idx
        data_examples.pred_pseudo_label = [
            [self.reasoner.idx_to_label[_idx] for _idx in sub_list] for sub_list in pred_idx
        ]
        return data_examples.pred_pseudo_label
    
    def filter_pseudo_label(self, data_examples):
        return super().filter_pseudo_label(data_examples)
    
    def abduce_pseudo_label(self, data_examples):
        return super().abduce_pseudo_label(data_examples)
        

    def _valid(self, data_examples: ListData, tag="") -> None:
        """
        Internal method for validating the model with given data examples.

        Parameters
        ----------
        data_examples : ListData
            Data examples to be used for validation.
        """
        self.predict(data_examples)
        self._valid_idx_to_pseudo_label(data_examples)

        for metric in self.metric_list:
            metric.process(data_examples)

        res = dict()
        for metric in self.metric_list:
            res.update(metric.evaluate())
        msg = "Evaluation ended, "
        for k, v in res.items():
            msg += k + f": {v:.3f} "
            
        wandb.log({f"{k}/{tag}": v for k, v in res.items()})
        
        print_log(msg, logger="current")
    
    def valid(self, val_data):
        return super().valid(val_data)
    
    