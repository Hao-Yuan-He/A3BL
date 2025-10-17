import argparse
import os.path as osp
from typing import List, Any
import torch
from torch import nn
from torch.optim import RMSprop, lr_scheduler

from ablkit.data.evaluation import ReasoningMetric, SymbolAccuracy
from ablkit.learning import  BasicNN, A3BLModel
from ablkit.reasoning import  A3BLReasoner, CachedKB, confidence_dist
from collections import defaultdict
from ablkit.utils import ABLLogger, print_log
from pathlib import Path

from datasets import (
    digits_to_number,
    get_ensemble_add,
)
from models.nn import LeNet5, ResNet50
from bridge import MixedBridge
import wandb


_ROOT = Path(__file__).parent


def split_list(lst):
    middle = len(lst) // 2
    list1 = lst[:middle]
    list2 = lst[middle:]
    return list1, list2


def parse_nums_and_ops(lst):
    op = lst[-1]
    list1, list2 = split_list(lst[:-1])
    return list1, list2, op


class MixedKB(CachedKB):
    def __init__(self, pseudo_label_list, GKB_len_list, max_err=1e-10, kb_file_path=None, mod1: int = 10, mod2: int = 10):
        self.mod1, self.mod2 = mod1, mod2
        super().__init__(pseudo_label_list, GKB_len_list, max_err, kb_file_path)

    def abduce_candidates(
        self,
        pseudo_label: List[Any],
        y: Any,
        x: List[Any],
        max_revision_num: int,
        require_more_revision: int,
    ) -> List[List[Any]]:

        candidates = self._find_candidate_GKB(pseudo_label, y)
        if len(candidates) == 0:
            return [], []
        return candidates, []

    def _valid_candidate(self, lst):
        if (len(lst) - 1) % 2 != 0:
            return False
        if lst[-1] not in [self.mod1, self.mod2]:
            return False
        return True

    def _find_candidate_GKB(self, pseudo_label, y):
        possible_candidate = self.GKB[len(pseudo_label)][y]
        possible_candidate = list(filter(lambda x: x[-1] == pseudo_label[-1], possible_candidate))
        return possible_candidate

    def logic_forward(self, lsts):
        if not self._valid_candidate(lsts):
            return None
        nums1, nums2, op = parse_nums_and_ops(lsts)
        nums1, nums2 = digits_to_number(nums1), digits_to_number(nums2)
        if nums1 >= (len(lsts) - 1) / 2 * 10 or nums2 >= (len(lsts) - 1) / 2 * 10:
            return None
        return (nums1 + nums2) % op


class MixedReasoner(A3BLReasoner):
    def __init__(
        self,
        kb,
        dist_func="confidence",
        idx_to_label=None,
        max_revision=-1,
        require_more_revision=0,
        use_zoopt=False,
        topK=16,
        temperature=0.2,
        multi_label=False,
    ):
        super().__init__(kb, dist_func, idx_to_label, max_revision, require_more_revision, use_zoopt, topK, temperature, multi_label)

    def abduce(self, data_example):
        return super().abduce(data_example)

    def _candidates_idxs(self, candidates: List[List[Any]]):
        return [[self.label_to_idx[x] for x in c[:-1]] for c in candidates]

    def abduce(self, data_example) -> List[Any]:
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

        confidence_dist_cal = confidence_dist

        candidate_probs = confidence_dist_cal(data_example.pred_prob, self._candidates_idxs(candidates), self.temperature)

        topk_candidates, topk_candidates_probs = self._topk(candidates, candidate_probs, self.topK)
        aggregated_labels = (
            self.aggregate(topk_candidates, topk_candidates_probs)[:-1]
            if not self.multi_label
            else self.multi_label_aggregate(topk_candidates, topk_candidates_probs)
        )
        return aggregated_labels, topk_candidates[0]
    
class MixedReasoningAccuracy(ReasoningMetric):
    def process(self, data_examples) -> None:
        pred_pseudo_label_list = data_examples.pred_pseudo_label
        y_list = data_examples.Y
        ops = data_examples.Op
        for pred_pseudo_label, y,  op in zip(pred_pseudo_label_list, y_list, ops):
            if self.kb._check_equal(
                self.kb.logic_forward(pred_pseudo_label + [op]), y
            ):
                self.results.append(1)
            else:
                self.results.append(0)


def main():
    parser = argparse.ArgumentParser(description="MNIST Addition example")
    parser.add_argument("--dataset", type=str, default="MNIST")
    parser.add_argument("--digit_size", type=int, default=1)
    parser.add_argument("--mod1", type=int, default=6)
    parser.add_argument("--mod2", type=int, default=8)
    parser.add_argument("--no-cuda", action="store_true", default=False, help="disables CUDA training")
    parser.add_argument("--epochs", type=int, default=1, help="number of epochs in each learning loop iteration (default : 1)")
    parser.add_argument("--label-smoothing", type=float, default=0.2, help="label smoothing in cross entropy loss (default : 0.2)")
    parser.add_argument("--lr", type=float, default=3e-4, help="base model learning rate (default : 0.0003)")
    parser.add_argument("--alpha", type=float, default=0.9, help="alpha in RMSprop (default : 0.9)")
    parser.add_argument("--batch-size", type=int, default=64, help="base model batch size (default : 32)")
    parser.add_argument("--segment_size", type=int, default=2048, help="segment size (default : 0.01)")
    parser.add_argument("--save_interval", type=int, default=1, help="save interval (default : 1)")
    parser.add_argument("--max-revision", type=int, default=-1, help="maximum revision in reasoner (default : -1)")
    parser.add_argument("--require-more-revision", type=int, default=2, help="require more revision in reasoner (default : 10)")
    parser.add_argument("--temp", type=float, default=0.2)
    parser.add_argument("--sample_size", type=int, default=30000)
    parser.add_argument("--topk", type=int, default=32, help="choose only top k candidates, k=-1 means use all of them.")
    args = parser.parse_args()

    wandb.init(project="LearnablityOfNeSy", group=f"addition {args.dataset}-{args.digit_size}-{args.mod1}-{args.mod2}")
    # Build logger
    print_log("Abductive Learning on the MNIST Addition example.", logger="current")

    # -- Working with Data ------------------------------
    print_log("Working with Data.", logger="current")
    train_data = get_ensemble_add(
        args.dataset, train=True, get_pseudo_label=True, n=args.digit_size, mod1=args.mod1, mod2=args.mod2, sample_size=args.sample_size
    )
    test_data = get_ensemble_add(args.dataset, train=False, get_pseudo_label=True, n=args.digit_size, mod1=args.mod1, mod2=args.mod2)

    # -- Building the Learning Part ---------------------
    print_log("Building the Learning Part.", logger="current")

    # Build necessary components for BasicNN
    cls_map = defaultdict(lambda: LeNet5(num_classes=10))
    cls_map.update(
        {"MNIST": LeNet5(num_classes=10), "KMNIST": LeNet5(num_classes=10), "CIFAR": ResNet50(num_classes=10), "SVHN": ResNet50(num_classes=10)}
    )

    cls = cls_map[args.dataset]

    loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = RMSprop(cls.parameters(), lr=args.lr, alpha=args.alpha)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    scheduler = lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        pct_start=0.15,
        epochs=args.epochs,
        steps_per_epoch=int(max(1 / args.segment_size, args.segment_size)),
    )

    # Build BasicNN
    base_model = BasicNN(
        cls,
        loss_fn,
        optimizer,
        scheduler=scheduler,
        device=device,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
    )

    # Build ABLModel
    model = A3BLModel(base_model)

    # -- Building the Reasoning Part --------------------
    print_log("Building the Reasoning Part.", logger="current")

    kb = MixedKB(
        pseudo_label_list=list(range(10)),
        GKB_len_list=[args.digit_size * 2 + 1],
        kb_file_path=f"{_ROOT}/kb_cache/addition_{args.digit_size}_mod1{args.mod1}_mod2{args.mod2}_kb",
        mod1=args.mod1,
        mod2=args.mod2,
    )

    # Create reasoner

    reasoner = MixedReasoner(
        kb, max_revision=args.max_revision, require_more_revision=args.require_more_revision, topK=args.topk, temperature=args.temp
    )

    # -- Building Evaluation Metrics --------------------
    print_log("Building Evaluation Metrics.", logger="current")
    metric_list = [SymbolAccuracy(prefix="mnist_add"), MixedReasoningAccuracy(kb=kb, prefix="mnist_add")]

    # -- Bridging Learning and Reasoning ----------------
    print_log("Bridge Learning and Reasoning.", logger="current")
    bridge = MixedBridge(model, reasoner, metric_list)

    # Retrieve the directory of the Log file and define the directory for saving the model weights.
    log_dir = ABLLogger.get_current_instance().log_dir
    weights_dir = osp.join(log_dir, "weights")

    #  Train and Test
    bridge.train(
        train_data,
        loops=args.epochs,
        segment_size=args.segment_size,
        save_interval=args.save_interval,
        save_dir=weights_dir,
    )
    bridge.test(test_data)


if __name__ == "__main__":
    main()
