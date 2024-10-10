from abl.bridge import SimpleBridge, WeaklySupervisedBridge
from abl.evaluation import ABLMetric, SymbolMetric
from abl.learning import ABLModel, BasicNN, WeaklySupervisedABLModel, WeaklySupervisedNN
from abl.reasoning import KBBase, ReasonerBase, WeaklySupervisedReasoner
from abl.utils import ABLLogger
from datasets.get_hwf import get_hwf
from models.nn import LeNet5
import numpy as np
import torch
import torch.nn as nn
import wandb
import argparse


class HWF_KB(KBBase):
    def __init__(
        self,
        pseudo_label_list=[
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "+",
            "-",
            "times",
            "div",
        ],
        prebuild_GKB=False,
        GKB_len_list=[1, 3, 5, 7],
        max_err=1e-3,
        use_cache=True,
    ):
        super().__init__(
            pseudo_label_list, prebuild_GKB, GKB_len_list, max_err, use_cache
        )

    def _valid_candidate(self, formula):
        if len(formula) % 2 == 0:
            return False
        for i in range(len(formula)):
            if i % 2 == 0 and formula[i] not in [
                "1",
                "2",
                "3",
                "4",
                "5",
                "6",
                "7",
                "8",
                "9",
            ]:
                return False
            if i % 2 != 0 and formula[i] not in ["+", "-", "times", "div"]:
                return False
        return True

    def logic_forward(self, formula):
        if not self._valid_candidate(formula):
            return None
        mapping = {str(i): str(i) for i in range(1, 10)}
        mapping.update({"+": "+", "-": "-", "times": "*", "div": "/"})
        formula = [mapping[f] for f in formula]
        return eval("".join(formula))


def main(args):
    logger = ABLLogger.get_instance("abl")
    wandb.init(project="ws_abl", group=f"HWF {args.dataset} ws abl")
    kb = HWF_KB(prebuild_GKB=True)
    abducer = WeaklySupervisedReasoner(kb, dist_func="confidence")
    cls = LeNet5(num_classes=len(kb.pseudo_label_list), image_size=(45, 45, 1))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss(reduction="none")
    optimizer = torch.optim.AdamW(cls.parameters(), lr=0.001, betas=(0.9, 0.99))
    base_model = WeaklySupervisedNN(
        cls,
        criterion,
        optimizer,
        device,
        save_interval=1,
        save_dir=logger.save_dir,
        batch_size=1024,
        num_epochs=1,
    )

    model = WeaklySupervisedABLModel(base_model, topK=args.topk)

    metric = [SymbolMetric(prefix="HWF"), ABLMetric(prefix="HWF")]
    train_data = get_hwf(train=True, get_pseudo_label=True)
    test_data = get_hwf(train=False, get_pseudo_label=True)

    bridge = WeaklySupervisedBridge(model, abducer, metric)
    bridge.train(
        train_data,
        epochs=args.epoches,
        batch_size=2048,
        more_revision=2,
        test_data=test_data,
    )


def get_args():
    parser = argparse.ArgumentParser(prog="HWF Experiment, Weakly Supervised ABL")
    parser.add_argument("--dataset", type=str, default="HWF")
    parser.add_argument("--epoches", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--topk",
        type=int,
        default=-1,
        help="choose only top k candidates, k=-1 means use all of them.",
    )
    args = parser.parse_args()
    return args


def seed_everything(seed: int = 0):
    import random, os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

if __name__ == '__main__':
    args = get_args()
    seed_everything(args.seed)
    main(args)