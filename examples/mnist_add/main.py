import argparse
import os.path as osp

import torch
from torch import nn
from torch.optim import RMSprop, lr_scheduler

from ablkit.bridge import SimpleBridge, A3BLBridge
from ablkit.data.evaluation import ReasoningMetric, SymbolAccuracy
from ablkit.learning import ABLModel, BasicNN, A3BLModel
from ablkit.reasoning import (
    KBBase,
    PrologKB,
    Reasoner,
    A3BLReasoner,
    CachedKB,
)
from collections import defaultdict
from ablkit.utils import ABLLogger, print_log
from pathlib import Path

from datasets import (
    get_mnist_add,
    digits_to_number,
    get_cifar_add,
    get_kmnist_add,
    get_svhn_add,
)
from models.nn import LeNet5, ResNet50

_ROOT = Path(__file__).parent


def split_list(lst):
    middle = len(lst) // 2
    list1 = lst[:middle]
    list2 = lst[middle:]
    return list1, list2


class AddKB(KBBase):
    def __init__(self, pseudo_label_list=list(range(10))):
        super().__init__(pseudo_label_list)

    def logic_forward(self, nums):
        return sum(nums)


class AddGroundKB(CachedKB):
    def __init__(self, pseudo_label_list=list(range(10)), GKB_len_list=[2], kb_file_path=""):
        super().__init__(pseudo_label_list, GKB_len_list, kb_file_path=kb_file_path)

    def logic_forward(self, nums):
        nums1, nums2 = split_list(nums)
        return digits_to_number(nums1) + digits_to_number(nums2)


MODEL = {"abl": ABLModel, "a3bl": A3BLModel}
BRIDGE = {"abl": SimpleBridge, "a3bl": A3BLBridge}
REASONER = {"abl": Reasoner, "a3bl": A3BLReasoner}


def main():
    parser = argparse.ArgumentParser(description="MNIST Addition example")
    parser.add_argument("--method", type=str, default="abl")
    parser.add_argument("--dataset", type=str, default="MNIST")
    parser.add_argument("--digit_size", type=int, default=1)
    parser.add_argument("--no-cuda", action="store_true", default=False, help="disables CUDA training")
    parser.add_argument("--epochs", type=int, default=1, help="number of epochs in each learning loop iteration (default : 1)")
    parser.add_argument("--label-smoothing", type=float, default=0.2, help="label smoothing in cross entropy loss (default : 0.2)")
    parser.add_argument("--lr", type=float, default=3e-4, help="base model learning rate (default : 0.0003)")
    parser.add_argument("--alpha", type=float, default=0.9, help="alpha in RMSprop (default : 0.9)")
    parser.add_argument("--batch-size", type=int, default=64, help="base model batch size (default : 32)")
    parser.add_argument("--loops", type=int, default=2, help="number of loop iterations (default : 2)")
    parser.add_argument("--segment_size", type=int, default=2048, help="segment size (default : 0.01)")
    parser.add_argument("--save_interval", type=int, default=1, help="save interval (default : 1)")
    parser.add_argument("--max-revision", type=int, default=-1, help="maximum revision in reasoner (default : -1)")
    parser.add_argument("--require-more-revision", type=int, default=2, help="require more revision in reasoner (default : 10)")
    kb_type = parser.add_mutually_exclusive_group()
    kb_type.add_argument("--prolog", action="store_true", default=False, help="use PrologKB (default: False)")
    kb_type.add_argument("--ground", action="store_true", default=False, help="use GroundKB (default: False)")
    parser.add_argument("--temp", type=float, default=0.2)
    parser.add_argument("--topk", type=int, default=32, help="choose only top k candidates, k=-1 means use all of them.")
    args = parser.parse_args()

    dta_map = {"MNIST": get_mnist_add, "KMNIST": get_kmnist_add, "CIFAR": get_cifar_add, "SVHN": get_svhn_add}

    # Build logger
    print_log("Abductive Learning on the MNIST Addition example.", logger="current")

    # -- Working with Data ------------------------------
    print_log("Working with Data.", logger="current")
    if args.dataset in dta_map:
        get_data = dta_map[args.dataset]
        train_data = get_data(train=True, get_pseudo_label=True, n=args.digit_size)
        test_data = get_data(train=False, get_pseudo_label=True, n=args.digit_size)
    else:
        raise NotImplementedError

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
        epochs=args.loops,
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
    model = MODEL[args.method](base_model)

    # -- Building the Reasoning Part --------------------
    print_log("Building the Reasoning Part.", logger="current")

    # Build knowledge base
    if args.prolog:
        kb = PrologKB(pseudo_label_list=list(range(10)), pl_file="add.pl")
    elif args.ground:
        kb = AddGroundKB(GKB_len_list=[args.digit_size * 2], kb_file_path=f"{_ROOT}/kb_cache/addition_{args.digit_size}_kb")
    else:
        kb = AddKB()

    # Create reasoner

    if args.method == "a3bl":
        reasoner = A3BLReasoner(
            kb, max_revision=args.max_revision, require_more_revision=args.require_more_revision, topK=args.topk, temperature=args.temp
        )
    else:
        reasoner = Reasoner(kb, max_revision=args.max_revision, require_more_revision=args.require_more_revision)

    # -- Building Evaluation Metrics --------------------
    print_log("Building Evaluation Metrics.", logger="current")
    metric_list = [SymbolAccuracy(prefix="mnist_add"), ReasoningMetric(kb=kb, prefix="mnist_add")]

    # -- Bridging Learning and Reasoning ----------------
    print_log("Bridge Learning and Reasoning.", logger="current")
    bridge = BRIDGE[args.method](model, reasoner, metric_list)

    # Retrieve the directory of the Log file and define the directory for saving the model weights.
    log_dir = ABLLogger.get_current_instance().log_dir
    weights_dir = osp.join(log_dir, "weights")

    #  Train and Test
    bridge.train(
        train_data,
        loops=args.loops,
        segment_size=args.segment_size,
        save_interval=args.save_interval,
        save_dir=weights_dir,
    )
    bridge.test(test_data)


if __name__ == "__main__":
    main()
