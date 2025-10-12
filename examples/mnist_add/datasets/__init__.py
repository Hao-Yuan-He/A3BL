from .addition import get_mnist_add, digits_to_number, get_cifar_add, get_svhn_add, get_kmnist_add

get_dataset = get_mnist_add 

__all__ = ["get_dataset", "get_mnist_data", "digits_to_number", "get_cifar_add", "get_svhn_add", "get_kmnist_add"]