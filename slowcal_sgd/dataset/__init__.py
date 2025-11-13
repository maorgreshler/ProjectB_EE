from .mnist import MNIST
from .cifar10 import CIFAR10


DATASET_REGISTRY = {
    'mnist': MNIST,
    'cifar10': CIFAR10
}