# %load train_utils.py
import numpy as np
#from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from IPython.display import clear_output


def collate_fn(batch):
    return tuple(zip(*batch))


def get_datasets(download=False, transform=None, test=True):
    transform = transform or transforms.Compose([                     
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = MNIST('.', train=True, download=download, transform=transform)
    if test:
        test_dataset = MNIST('.', train=False, transform=transform)

    return train_dataset, test_dataset if test else train_dataset


def get_loaders(download=False, new_transform=None, batch_size=32):
    transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                        ])
    train_dataset, test_dataset = get_datasets(download)

    if new_transform:
        new_train_dataset = get_datasets(download=True, transform=new_transform, test=False)
        train_dataset = train_dataset + new_train_dataset

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)# , collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)# , collate_fn=collate_fn)

    return train_loader, test_loader


def _epoch(network, loss, loader,
           backward=True,
           optimizer=None,
           device='cpu',
           ravel_init=False):
    losses = []
    accuracies = []
    for X, y in loader:
        X = X.to(device)
        y = y.to(device)
        if ravel_init:
            X = X.view(X.size(0), -1)
        network.zero_grad()
        prediction = network(X)
        loss_batch = loss(prediction, y)
        losses.append(loss_batch.cpu().item())
        if backward:
            loss_batch.backward()
            optimizer.step()
        prediction = prediction.max(1)[1]
        accuracies.append((prediction == y).cpu().float().numpy().mean())
    return losses, accuracies


def train(network, train_loader=None, test_loader=None, epochs=10, 
          learning_rate=1e-3, plot=True, verbose=True, loss=None, 
          optimizer=None, clear_data=True, get_loaders_func=None,
          ravel_init=False, device='cpu', tolerate_keyboard_interrupt=True):
    loss = loss() if loss else nn.NLLLoss()
    optimizer = optimizer(network.parameters(), learning_rate) if optimizer else torch.optim.Adam(network.parameters(), lr=learning_rate)
    if train_loader is None and get_loaders_func is None:
        raise RuntimeError("No train_loader")

    train_loss_epochs = []
    test_loss_epochs = []
    train_accuracy_epochs = []
    test_accuracy_epochs = []
    network = network.to(device)
    try:
        for epoch in range(epochs):
            if get_loaders_func:
                train_loader, test_loader = get_loaders_func()
            if train_loader:
                network.train()
                losses, accuracies = _epoch(network,
                                            loss,
                                            train_loader,
                                            True,
                                            optimizer,
                                            device,
                                            ravel_init)
                train_loss_epochs.append(np.mean(losses))
                train_accuracy_epochs.append(np.mean(accuracies))

            if test_loader:
                network.eval()
                losses, accuracies = _epoch(network,
                                            loss,
                                            test_loader,
                                            False,
                                            optimizer,
                                            device,
                                            ravel_init)

                test_loss_epochs.append(np.mean(losses))
                test_accuracy_epochs.append(np.mean(accuracies))
            if verbose:
                if clear_data:
                    clear_output(True)
                if test_loader:
                    print('Epoch {0}... (Train/Test) Loss: {1:.3f}/{2:.3f}\tAccuracy: {3:.3f}/{4:.3f}'.format(
                            epoch, train_loss_epochs[-1], test_loss_epochs[-1],
                            train_accuracy_epochs[-1], test_accuracy_epochs[-1]))
                else:
                    print('Epoch {0}... (Train) Loss: {1:.3f}\tAccuracy: {2:.3f}'.format(
                            epoch, train_loss_epochs[-1], train_accuracy_epochs[-1]))
            if plot:
                plt.figure(figsize=(12, 5))
                plt.subplot(1, 2, 1)
                plt.plot(train_loss_epochs, label='Train')
                if test_loader:
                    plt.plot(test_loss_epochs, label='Test')
                plt.xlabel('Epochs', fontsize=16)
                plt.ylabel('Loss', fontsize=16)
                plt.legend(loc=0, fontsize=16)
                plt.grid()
                plt.subplot(1, 2, 2)
                plt.plot(train_accuracy_epochs, label='Train accuracy')
                if test_loader:
                    plt.plot(test_accuracy_epochs, label='Test accuracy')
                plt.xlabel('Epochs', fontsize=16)
                plt.ylabel('Accuracy', fontsize=16)
                plt.legend(loc=0, fontsize=16)
                plt.grid()
                plt.show()
    except KeyboardInterrupt:
        if tolerate_keyboard_interrupt:
            pass
        else:
            raise KeyboardInterrupt
    return train_loss_epochs, \
           test_loss_epochs, \
           train_accuracy_epochs, \
           test_accuracy_epochs


def plot_comp(test_loss, test_accuracy, name_start='', name_end=''):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.title('Loss')
    for name in test_loss:
        if name.startswith(name_start) and name.endswith(name_end):
            plt.plot(test_loss[name], label=name)
    plt.xlabel('Epochs', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.legend(loc=0, fontsize=16)
    plt.grid()
    plt.subplot(1, 2, 2)
    plt.title('Accuracy')
    for name in test_accuracy:
        if name.startswith(name_start) and name.endswith(name_end):
            plt.plot(test_accuracy[name], label=name)
    plt.xlabel('Epochs', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.legend(loc=0, fontsize=16)
    plt.grid()
    plt.show()


def plot_analysis(network):
    wrong_X = []
    correct_y = []
    predicted_y = []
    logits = []
    for X, y in test_loader:
        prediction = network(X)
        prediction = np.exp(prediction.data.numpy())
        prediction /= prediction.sum(1, keepdims=True)
        for i in range(len(prediction)):
            if np.argmax(prediction[i]) != y[i]:
                wrong_X.append(X[i])
                correct_y.append(y[i])
                predicted_y.append(np.argmax(prediction[i]))
                logits.append(prediction[i][y[i]])
    wrong_X = np.row_stack(wrong_X)
    correct_y = np.row_stack(correct_y)[:, 0]
    predicted_y = np.row_stack(predicted_y)[:, 0]
    logits = np.row_stack(logits)[:, 0]

    plt.figure(figsize=(10, 5))
    order = np.argsort(logits)
    for i in range(21):
        plt.subplot(3, 7, i+1)
        plt.imshow(wrong_X[order[i]].reshape(28, 28), cmap=plt.cm.Greys_r)
        plt.title('{}({})'.format(correct_y[order[i]], predicted_y[order[i]]), fontsize=20)
        plt.axis('off')
