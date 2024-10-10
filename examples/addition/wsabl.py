import torch.nn as nn
import torch

from abl.reasoning import ReasonerBase, KBBase, WeaklySupervisedReasoner
from pathlib import Path
from abl.learning import BasicNN, ABLModel, WeaklySupervisedABLModel, WeaklySupervisedNN
from abl.bridge import SimpleBridge, WeaklySupervisedBridge
from abl.evaluation import SymbolMetric, ABLMetric
from abl.utils import ABLLogger
from collections import defaultdict
from models.nn import LeNet5, ResNet50
import argparse
import wandb
from datasets.addition import get_mnist_add, digits_to_number, get_cifar_add, get_kmnist_add, get_svhn_add

_ROOT = Path(__file__).parent
def split_list(lst):
    middle = len(lst) // 2
    list1 = lst[:middle]
    list2 = lst[middle:]
    return list1, list2


class add_KB(KBBase):
    def __init__(
        self,
        pseudo_label_list=list(range(10)),
        prebuild_GKB=False,
        GKB_len_list=[1 * 2],
        max_err=0,
        use_cache=True,
        kb_file_path='',
    ):
        super().__init__(
            pseudo_label_list, prebuild_GKB, GKB_len_list, max_err, use_cache, kb_file_path
        )

    def logic_forward(self, nums):
        nums1, nums2 = split_list(nums)
        return digits_to_number(nums1) + digits_to_number(nums2)


def main(args):
    logger = ABLLogger.get_instance("abl")
    wandb.init(
        project="ws_abl",
        group=f"{args.group_hint}: addition {args.dataset}-{args.digit_size} ws abl",
    )
    seed_everything(args.seed)
    kb = add_KB(prebuild_GKB=True, GKB_len_list=[args.digit_size * 2], kb_file_path=f"{_ROOT}/kb_cache/addition_{args.digit_size}_kb")
    abducer = WeaklySupervisedReasoner(kb)
    cls_map = defaultdict(lambda: LeNet5(
        num_classes=len(kb.pseudo_label_list)))
    cls_map.update({"MNIST": LeNet5(num_classes=len(kb.pseudo_label_list))})
    cls_map.update({"KMNIST": LeNet5(num_classes=len(kb.pseudo_label_list))})
    cls_map.update({"CIFAR": ResNet50(num_classes=len(kb.pseudo_label_list))})
    cls_map.update({"SVHN": ResNet50(num_classes=len(kb.pseudo_label_list))})
    cls = cls_map[args.dataset]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss(reduction="none")
    optimizer = torch.optim.Adam(cls.parameters(), lr=0.001, betas=(0.9, 0.99))
    base_model = WeaklySupervisedNN(
        cls,
        criterion,
        optimizer,
        device,
        save_interval=1,
        save_dir=logger.save_dir,
        batch_size=256,
        num_epochs=1,
    )
    model = WeaklySupervisedABLModel(
        base_model, topK=args.topk, temp=args.temp)
    metric = [
        SymbolMetric(prefix=f"{args.dataset}_add"),
        ABLMetric(prefix=f"{args.dataset}_add"),
    ]
    dta_map = defaultdict(None)
    dta_map.update({"MNIST": get_mnist_add})
    dta_map.update({"KMNIST": get_kmnist_add})
    dta_map.update({"CIFAR": get_cifar_add})
    dta_map.update({"SVHN": get_svhn_add})
    get_data = dta_map[args.dataset]
    train_data = get_data(train=True, get_pseudo_label=True, n=args.digit_size)
    test_data = get_data(train=False, get_pseudo_label=True, n=args.digit_size)
    bridge = WeaklySupervisedBridge(model, abducer, metric)
    bridge.train(
        train_data,
        epochs=args.epoches,
        batch_size=1024,
        test_data=test_data,
        more_revision=2,
    )
    # bridge.test(test_data)


def get_args():
    parser = argparse.ArgumentParser(
        prog="Addition Experiment, Weakly Supervised ABL")
    parser.add_argument("--group_hint", type=str, default="")
    parser.add_argument("--dataset", type=str, default="MNIST")
    parser.add_argument("--digit_size", type=int, default=1)
    parser.add_argument("--epoches", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temp", type=float, default=0.2)
    parser.add_argument(
        "--topk",
        type=int,
        default=32,
        help="choose only top k candidates, k=-1 means use all of them.",
    )
    args = parser.parse_args()
    return args


def seed_everything(seed: int = 0):
    import random
    import os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


if __name__ == "__main__":
    args = get_args()
    main(args)
