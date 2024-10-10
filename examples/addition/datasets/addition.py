import itertools
import json
import random
from pathlib import Path
from typing import Callable, List, Iterable, Tuple

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset as TorchDataset
from collections import Counter, defaultdict

_DATA_ROOT = Path(__file__).parent


class ShuffleIterator:
    def __init__(self, iterable):
        self.iterable = iterable
        self.buffer = []
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.buffer):
            self.buffer = list(self.iterable)
            random.shuffle(self.buffer)
            self.index = 0

        if not self.buffer:
            self.buffer = list(self.iterable)
            random.shuffle(self.buffer)

        item = self.buffer[self.index]
        self.index += 1
        return item


def MNIST_datasets():
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    datasets = {
        "train": torchvision.datasets.MNIST(
            root=str(_DATA_ROOT), train=True, download=True, transform=transform
        ),
        "test": torchvision.datasets.MNIST(
            root=str(_DATA_ROOT), train=False, download=True, transform=transform
        ),
    }
    return datasets


def KMNIST_datasets():
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    datasets = {
        "train": torchvision.datasets.KMNIST(
            root=str(_DATA_ROOT), train=True, download=True, transform=transform
        ),
        "test": torchvision.datasets.KMNIST(
            root=str(_DATA_ROOT), train=False, download=True, transform=transform
        ),
    }
    return datasets


def CIFAR_datasets():
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    datasets = {
        "train": torchvision.datasets.cifar.CIFAR10(
            root=str(_DATA_ROOT), train=True, download=True, transform=transform
        ),
        "test": torchvision.datasets.cifar.CIFAR10(
            root=str(_DATA_ROOT), train=False, download=True, transform=transform
        ),
    }
    return datasets


def SVHN_datasets():
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    datasets = {
        "train": torchvision.datasets.SVHN(
            root=str(_DATA_ROOT), split='train', download=True, transform=transform
        ),
        "test": torchvision.datasets.SVHN(
            root=str(_DATA_ROOT), split='test', download=True, transform=transform
        ),
    }
    return datasets


def get_datasets(dataset_name: str = "MNIST"):
    if dataset_name == "MNIST":
        return MNIST_datasets()
    elif dataset_name == 'KMNIST':
        return KMNIST_datasets()
    elif dataset_name in ["CIFAR10", 'CIFAR']:
        return CIFAR_datasets()
    elif dataset_name == 'SVHN':
        return SVHN_datasets()


def digits_to_number(digits: Iterable[int]) -> int:
    number = 0
    for d in digits:
        number *= 10
        number += d
    return number


def addition(n: int, dataset: str, seed=None, train: bool = True):
    """Returns a dataset for binary addition"""
    return DigitsOperator(
        dataset_name=dataset,
        function_name="addition" if n == 1 else "multi_addition",
        operator=sum,
        size=n,
        arity=2,
        seed=seed,
        train=train,
    )


class DigitsOperator(TorchDataset):
    def __getitem__(self, index: int) -> Tuple[list, list, int]:
        l1, l2 = self.data[index]
        label = self._get_label(index)
        l1 = [self.dataset[x][0] for x in l1]
        l2 = [self.dataset[x][0] for x in l2]
        return l1, l2, label

    def balance_indices(self):
        balance_size = sorted(Counter(self.dataset.labels).items())[0][1]
        labels_dist = defaultdict(int)
        sampler_iter = ShuffleIterator(list(range(len(self.dataset))))
        balanced_indices = []
        while len(balanced_indices) < balance_size * 10:
            sample = next(sampler_iter)
            sampled_class = self.dataset.labels[sample]
            if labels_dist[sampled_class] >= balance_size:
                continue
            balanced_indices.append(sample)
            labels_dist[sampled_class] += 1
        return balanced_indices

    def indices(self):
        return list(range(len(self.dataset)))

    def __init__(
        self,
        dataset_name: str,
        function_name: str,
        operator: Callable[[List[int]], int],
        size=1,
        arity=2,
        seed=None,
        train: bool = True,
        sequence_num: int = 30000,
    ):
        """Generic dataset for operator(img, img) style datasets.

        :param dataset_name: Dataset to use (train, val, test)
        :param function_name: Name of Problog function to query.
        :param operator: Operator to generate correct examples
        :param size: Size of numbers (number of digits)
        :param arity: Number of arguments for the operator
        :param seed: Seed for RNG
        """
        super(DigitsOperator, self).__init__()
        assert size >= 1
        assert arity >= 1
        self.dataset_name = dataset_name
        self.datasets = get_datasets(dataset_name)
        self.dataset = self.datasets["train" if train else "test"]
        self.function_name = function_name
        self.operator = operator
        self.size = size
        self.arity = arity
        self.seed = seed
        mnist_indices = self.indices()
        
        if seed is not None:
            rng = random.Random(seed)
            rng.shuffle(mnist_indices)
        dataset_iter = ShuffleIterator(mnist_indices)
        # Build list of examples (mnist indices)
        self.data = []
        try:
            while len(self.data) < sequence_num:
                self.data.append(
                    [
                        [next(dataset_iter) for _ in range(self.size)]
                        for _ in range(self.arity)
                    ]
                )
        except StopIteration:
            pass

    def to_file_repr(self, i):
        """Old file represenation dump. Not a very clear format as multi-digit arguments are not separated"""
        return f"{tuple(itertools.chain(*self.data[i]))}\t{self._get_label(i)}"

    def to_json(self):
        """
        Convert to JSON, for easy comparisons with other systems.

        Format is [EXAMPLE, ...]
        EXAMPLE :- [ARGS, expected_result]
        ARGS :- [MULTI_DIGIT_NUMBER, ...]
        MULTI_DIGIT_NUMBER :- [mnist_img_id, ...]
        """
        data = [(self.data[i], self._get_label(i)) for i in range(len(self))]
        return json.dumps(data)

    def _get_label(self, i: int):
        mnist_indices = self.data[i]
        # Figure out what the ground truth is, first map each parameter to the value:
        ground_truth = [
            digits_to_number(self.dataset[j][1] for j in i) for i in mnist_indices
        ]
        # Then compute the expected value:
        expected_result = self.operator(ground_truth)
        return expected_result

    def _get_symbol_label(self, i: int):
        mnist_indices = self.data[i]
        # Figure out what the ground truth is, first map each parameter to the value:
        ground_truth = [self.dataset[j][1] for i in mnist_indices for j in i]
        return ground_truth

    def __len__(self):
        return len(self.data)


def get_mnist_add(train=True, get_pseudo_label=False, n=1):
    mnistDataset = addition(n, "MNIST", train=train)
    X, Y, Z = [], [], []
    for idx in range(len(mnistDataset)):
        x1, x2, y = mnistDataset[idx]
        z = mnistDataset._get_symbol_label(idx)
        X.extend([x1 + x2]), Y.append(y), Z.extend([z])
    if get_pseudo_label:
        return X, Z, Y
    return X, None, Y


def get_kmnist_add(train=True, get_pseudo_label=False, n=1):
    mnistDataset = addition(n, "KMNIST", train=train)
    X, Y, Z = [], [], []
    for idx in range(len(mnistDataset)):
        x1, x2, y = mnistDataset[idx]
        z = mnistDataset._get_symbol_label(idx)
        X.extend([x1 + x2]), Y.append(y), Z.extend([z])
    if get_pseudo_label:
        return X, Z, Y
    return X, None, Y


def get_cifar_add(train=True, get_pseudo_label=False, n=1):
    mnistDataset = addition(n, "CIFAR10", train=train)
    X, Y, Z = [], [], []
    for idx in range(len(mnistDataset)):
        x1, x2, y = mnistDataset[idx]
        z = mnistDataset._get_symbol_label(idx)
        X.extend([x1 + x2]), Y.append(y), Z.extend([z])
    if get_pseudo_label:
        return X, Z, Y
    return X, None, Y


def get_svhn_add(train=True, get_pseudo_label=False, n=1):
    mnistDataset = addition(n, "SVHN", train=train)
    X, Y, Z = [], [], []
    for idx in range(len(mnistDataset)):
        x1, x2, y = mnistDataset[idx]
        z = mnistDataset._get_symbol_label(idx)
        X.extend([x1 + x2]), Y.append(y), Z.extend([z])
    if get_pseudo_label:
        return X, Z, Y
    return X, None, Y


if __name__ == "__main__":
    mnist_add = get_mnist_add()
