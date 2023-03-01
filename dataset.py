import numpy as np
import torch

from tensorflow.keras.datasets import mnist, fashion_mnist
from torch.utils.data import DataLoader


def generate(digit, batch_size=128, dataset="MNIST"):
    # Load the dataset
    if dataset == "MNIST":
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
    if dataset == "FMNIST":
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    # Normalize the data
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Prepare the datasets
    x_train_normal = x_train[np.where(y_train == digit)[0]]
    train_len = int(0.9 * np.shape(x_train_normal)[0])
    x_train = np.reshape(x_train_normal[:train_len], (-1, 1, 28, 28))
    x_val = np.reshape(x_train_normal[train_len:], (-1, 1, 28, 28))

    # Get the data loaders
    train_loader = DataLoader(x_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(x_val, batch_size=batch_size)

    x_test = np.reshape(x_test, (-1, 1, 28, 28))
    normal_idx = np.where(y_test == digit)[0]
    anomalous_idx = np.where(y_test != digit)[0]
    test_indices = np.append(
        normal_idx,
        np.random.choice(anomalous_idx, size=len(normal_idx), replace=False),
    )
    test_loader = DataLoader(x_test[test_indices], batch_size=batch_size)
    y_true = y_test[test_indices]
    y_true = y_true == digit

    if digit == 0:
        print("Train:", len(x_train))
        print("Validation:", len(x_val))
        print("Testing:", len(test_indices))

    return train_loader, val_loader, test_loader, y_true
