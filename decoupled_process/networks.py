# This code is part of the EMViT-DDPM project and was authored by Victor Barreiro.

import torch.nn as nn


class CNN21(nn.Module):
    """
    A 2D Convolutional Neural Network for hyperspectral image classification.
    This model consists of two convolutional layers followed by two fully connected layers.

    Args:
        N1 (int): Number of input channels (bands).
        N2 (int): Number of output channels for the first convolutional layer.
        N3 (int): Number of output channels for the second convolutional layer.
        N4 (int): Number of input features for the first fully connected layer.
        N5 (int): Number of output classes.
        D1 (int): Stride for the first max-pooling layer.
        D2 (int): Stride for the second max-pooling layer.
    """

    def __init__(self, N1, N2, N3, N4, N5, D1, D2):
        super(CNN21, self).__init__()
        self.conv1 = nn.Conv2d(N1, N2, 3, padding=1)
        self.conv2 = nn.Conv2d(N2, N3, 3, padding=1)
        self.fc1 = nn.Linear(N4, 256)
        self.fc2 = nn.Linear(256, N5)
        self.D1 = D1
        self.D2 = D2

    def forward(self, x):
        """
        Defines the forward pass of the CNN.

        Args:
            x (torch.Tensor): The input tensor of shape (N, C, H, W).

        Returns:
            torch.Tensor: The output logits for each class.
        """
        # First convolutional layer with ReLU and max pooling
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.max_pool2d(x, self.D1, self.D1)

        # Second convolutional layer with ReLU and max pooling
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, self.D2, self.D2)

        # Flatten the tensor for the fully connected layers
        x = x.view(x.size(0), -1)

        # First fully connected layer with ReLU
        x = nn.functional.relu(self.fc1(x))

        # Output fully connected layer
        x = self.fc2(x)

        return x


# This code is part of the EMViT-DDPM project and was authored by Victor Barreiro.

# -*- coding: utf-8 -*-
"""
This script provides a set of utility functions for training and evaluating deep learning models,
primarily using PyTorch and the `timm` library. It includes functions for:
- Training loop with validation
- Model testing and accuracy calculation (including OA, AA, Kappa)
- Loss function selection (Cross-Entropy, Balanced CE, Focal Loss)
- Learning rate updates
- Plotting training history (loss and accuracy curves)

@author: Victor Barreiro
"""

import torch
import torch.nn as nn
from sklearn import preprocessing
import torchvision.transforms as transforms
import sklearn.utils.class_weight as class_weight
import torch.nn.functional as F
from einops import repeat
from focal_loss.focal_loss import FocalLoss
import math, random, time
import numpy as np
import timm
from sklearn.metrics import cohen_kappa_score, f1_score
import matplotlib.pyplot as plt

# Import the accelerated test function
from test_function import test


def update_lr(optimizer, lr):
    """
    Manually updates the learning rate of a PyTorch optimizer.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer whose learning rate will be updated.
        lr (float): The new learning rate to set.
    """
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def accuracy_mean_deviation(OA, AA, aa):
    """
    Calculates and prints the mean and standard deviation of model performance metrics
    over multiple experiments.

    Args:
        OA (list): A list of Overall Accuracy (OA) scores from each experiment.
        AA (list): A list of Average Accuracy (AA) scores from each experiment.
        aa (list of lists): A list containing lists of per-class accuracies for each experiment.
    """
    n = len(OA)
    nclases = len(aa[0])
    print(f"* Means and deviations ({n} exp):")

    # Calculate means
    OAm = sum(OA) / n
    AAm = sum(AA) / n
    aam = [0] * nclases
    for i in range(n):
        for j in range(1, nclases):
            aam[j] += aa[i][j]
    for j in range(1, nclases):
        aam[j] /= n

    # Calculate standard deviations
    OAd = 0
    AAd = 0
    aad = [0] * nclases
    for i in range(n):
        OAd += (OA[i] - OAm) ** 2
        AAd += (AA[i] - AAm) ** 2
        for j in range(1, nclases):
            aad[j] += (aa[i][j] - aam[j]) ** 2
    OAd = math.sqrt(OAd / (n - 1))
    AAd = math.sqrt(AAd / (n - 1))
    for j in range(1, nclases):
        aad[j] = math.sqrt(aad[j] / (n - 1))

    for j in range(1, nclases):
        print(f"    Class {j:02d}: {aam[j]:02.02f}+{aad[j]:02.02f}")
    print(f"    OA={OAm:02.02f}+{OAd:02.02f}, AA={AAm:02.02f}+{AAd:02.02f}")


def select_loss(str_loss, truth, device, n_classes):
    """
    Selects and configures a loss function based on a string identifier.

    Args:
        str_loss (str): Identifier for the loss function ('CE', 'balanced_CE', 'focal_class').
        truth (list or np.array): Ground truth labels, used for calculating weights in
                                  the balanced cross-entropy case.
        device (torch.device): The device to move the loss weights to.
        n_classes (int): The total number of classes.

    Returns:
        torch.nn.Module: The selected loss function instance.
    """
    if str_loss.lower() == "ce":
        return nn.CrossEntropyLoss()
    if str_loss.lower() == "balanced_ce":
        truth_no_zeros = np.array([x for x in truth if x != 0])
        all_classes = np.array(range(1, n_classes + 1))
        class_weights = np.ones_like(all_classes, dtype=np.float32)
        unique_classes = np.unique(truth_no_zeros)
        calculated_weights = class_weight.compute_class_weight(
            class_weight="balanced", classes=np.unique(truth_no_zeros), y=truth_no_zeros
        )
        class_weights[np.isin(all_classes, unique_classes)] = calculated_weights
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
        return nn.CrossEntropyLoss(weight=class_weights)
    if str_loss == "focal_class":
        return FocalLoss(alpha=0.5, gamma=2.0, reduction="mean")


def train(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    n_epochs,
    model_path="./best_model.pt",
):
    """
    Trains a model for a specified number of epochs, with validation after each epoch.
    Saves the model with the best validation accuracy.

    Args:
        model (torch.nn.Module): The model to be trained.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        val_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        optimizer (torch.optim.Optimizer): The optimizer for training.
        device (torch.device): The device (CPU or GPU) to perform training on.
        n_epochs (int): The total number of epochs for training.
        model_path (str, optional): Path to save the best performing model. Defaults to "./best_model.pt".

    Returns:
        tuple: A tuple containing lists of (training_losses, validation_losses,
               training_accuracies, validation_accuracies) for each epoch.
    """
    loss_fn = torch.nn.CrossEntropyLoss()
    acc_fn = lambda logit, label: torch.mean((logit.argmax(dim=-1) == label).float())

    training_losses = []
    training_acces = []
    validation_losses = []
    validation_acces = []

    best_val_acc = 0
    optimizer.zero_grad()
    for e in range(n_epochs):
        model.train()
        losses = []
        acces = []
        for i, (img, label) in enumerate(train_loader):
            img = img.to(device)
            label = label.to(device)
            logits = model(img)
            loss = loss_fn(logits, label)
            acc = acc_fn(logits, label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())
            acces.append(acc.item())

        # If a scheduler is attached to the optimizer, step it
        if hasattr(optimizer, "scheduler"):
            optimizer.scheduler.step()

        avg_train_loss = sum(losses) / len(losses)
        avg_train_acc = sum(acces) / len(acces)
        training_losses.append(avg_train_loss)
        training_acces.append(avg_train_acc)

        print(
            f"* Epoch {e}, Training loss: {avg_train_loss:.4f}, Acc: {avg_train_acc:.4f}."
        )

        model.eval()
        with torch.no_grad():
            losses = []
            acces = []
            for i, (img, label) in enumerate(val_loader):
                img = img.to(device)
                label = label.to(device)
                logits = model(img)
                loss = loss_fn(logits, label)
                acc = acc_fn(logits, label)
                losses.append(loss.item())
                acces.append(acc.item())
            avg_val_loss = sum(losses) / len(losses)
            avg_val_acc = sum(acces) / len(acces)
            validation_losses.append(avg_val_loss)
            validation_acces.append(avg_val_acc)
            print(f"    Validation loss: {avg_val_loss:.4f}, Acc: {avg_val_acc:.4f}.")

        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            print(f"    Saving best model with acc {best_val_acc:.4f} at epoch {e}!")
            torch.save(model.state_dict(), model_path)

    return training_losses, validation_losses, training_acces, validation_acces


import test_function as test

# def test(model, test_loader, height, width, device, truth, seg, center, train_set, validation_set, test_set, nclases, nclases_NOvacias, show=True):
#     """
#     NOTE: This is the pure Python version of the test function. For faster execution,
#     it is recommended to use the Cython-accelerated version imported from `test_function`.
#     This function should only be used if the Cython version is unavailable.
#
#     Tests the model's performance on a test set, considering an internal data representation
#     that uses superpixel segmentation. It calculates various metrics like Overall Accuracy (OA),
#     Average Accuracy (AA), per-class accuracy, Kappa coefficient, and F1-score.
#
#     Args:
#         model (torch.nn.Module): The trained model to be evaluated.
#         test_loader (torch.utils.data.DataLoader): DataLoader for the test set.
#         height (int): Height of the original image.
#         width (int): Width of the original image.
#         device (torch.device): The device to perform inference on.
#         truth (np.array): The ground truth map for the entire image.
#         seg (np.array): The superpixel segmentation map.
#         center (np.array): An array mapping segment index to its center pixel index.
#         train_set (list): List of indices used for training.
#         validation_set (list): List of indices used for validation.
#         test_set (list): List of indices used for testing.
#         nclases (int): Total number of classes.
#         nclases_NOvacias (int): Number of non-empty classes in the ground truth.
#         show (bool, optional): Whether to print progress and results. Defaults to True.
#
#     Returns:
#         tuple: A tuple containing OA, AA, per-class accuracies, Kappa, and F1-score.
#     """
#     dataset_test = test_loader.dataset
#     output = np.zeros(height * width, dtype=np.uint8)
#     model.eval()
#     start_time = time.time()
#     with torch.no_grad():
#         total = 0
#         for (inputs, labels) in test_loader:
#             inputs = inputs.to(device)
#             outputs = model(inputs)
#             (_, predicted) = torch.max(outputs.data, 1)
#             predicted_cpu = predicted.cpu()
#             for i in range(len(predicted_cpu)):
#                 # Classes are 1-indexed
#                 output[test_set[total + i]] = np.uint8(predicted_cpu[i] + 1)
#             total += labels.size(0)
#             if show and (total % 2000 == 0):
#                 print(f'    Test: {total:6d}/{len(dataset_test)}')
#
#     if show:
#         print('* Generating classification map...')
#
#     for i in range(height * width):
#         output[i] = output[center[seg[i]]]
#
#     # Exclude training and validation samples from the output map
#     for i in train_set: output[i] = 0
#     for i in validation_set: output[i] = 0
#
#     correct = 0
#     total = 0
#     for i in range(len(center)):
#         if output[center[i]] == 0: continue
#         total += 1
#         if output[center[i]] == truth[center[i]]:
#             correct += 1
#     acc = 100 * correct / total
#     if show:
#         print(f'* Accuracy (segments): {acc:.02f}')
#
#     # Pixel-level precision
#     correct = 0
#     total = 0
#     AA = 0
#     class_correct = [0] * (nclases + 1)
#     class_total = [0] * (nclases + 1)
#     class_aa = [0] * (nclases + 1)
#
#     for i in range(len(output)):
#         if output[i] == 0 or truth[i] == 0: continue
#         total += 1
#         class_total[truth[i]] += 1
#         if output[i] == truth[i]:
#             correct += 1
#             class_correct[truth[i]] += 1
#
#     for i in range(1, nclases + 1):
#         if class_total[i] != 0:
#             class_aa[i] = 100 * class_correct[i] / class_total[i]
#         else:
#             class_aa[i] = 0
#         AA += class_aa[i]
#
#     OA = 100 * correct / total
#     AA = AA / nclases_NOvacias
#
#     # Remove zero-labeled pixels for Kappa and F1 score calculation
#     non_zero_indices = np.where((output != 0) & (truth != 0))
#     output_flat = output[non_zero_indices]
#     truth_flat = truth[non_zero_indices]
#
#     kappa = cohen_kappa_score(truth_flat, output_flat)
#     f1 = f1_score(truth_flat, output_flat, average='weighted')
#
#     if show:
#         print('* Accuracy (pixels)')
#         for i in range(1, nclases + 1):
#             print(f'    Class {i:02d}: {class_aa[i]:02.02f}')
#         print(f'* Accuracy (pixels), OA={OA:02.02f}, AA={AA:02.02f}')
#         print(f'    total: {total}, correct: {correct}')
#
#     return (OA, AA, class_aa, kappa, f1)


def test_without_segments(
    model,
    test_loader,
    height,
    width,
    device,
    truth,
    test_set,
    nclases,
    nclases_NOvacias,
    show=True,
):
    """
    Tests the model's performance on a test set without using superpixel segmentation.
    It directly evaluates the classification of individual pixels.

    Args:
        model (torch.nn.Module): The trained model to be evaluated.
        test_loader (torch.utils.data.DataLoader): DataLoader for the test set.
        height (int): Height of the original image.
        width (int): Width of the original image.
        device (torch.device): The device to perform inference on.
        truth (np.array): The ground truth map for the entire image.
        test_set (list): List of indices used for testing.
        nclases (int): Total number of classes.
        nclases_NOvacias (int): Number of non-empty classes in the ground truth.
        show (bool, optional): Whether to print progress and results. Defaults to True.

    Returns:
        tuple: A tuple containing OA, AA, per-class accuracies, Kappa, and F1-score.
    """
    output = np.zeros(height * width, dtype=np.uint8)
    model.eval()
    with torch.no_grad():
        total = 0
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            (_, predicted) = torch.max(outputs.data, 1)
            predicted_cpu = predicted.cpu()
            for i in range(len(predicted_cpu)):
                # Classes are 1-indexed
                output[test_set[total + i]] = np.uint8(predicted_cpu[i] + 1)
            total += labels.size(0)
            if show and (total % 2000 == 0):
                print(f"    Test: {total:6d}/{len(test_loader.dataset)}")

    if show:
        print("* Generating classification map...")

    correct = 0
    total = 0
    AA = 0
    class_correct = [0] * (nclases + 1)
    class_total = [0] * (nclases + 1)
    class_aa = [0] * (nclases + 1)

    for i in range(len(output)):
        if output[i] == 0 or truth[i] == 0:
            continue
        total += 1
        class_total[truth[i]] += 1
        if output[i] == truth[i]:
            correct += 1
            class_correct[truth[i]] += 1

    for i in range(1, nclases + 1):
        if class_total[i] != 0:
            class_aa[i] = 100 * class_correct[i] / class_total[i]
        else:
            class_aa[i] = 0
        AA += class_aa[i]

    OA = 100 * correct / total
    AA = AA / nclases_NOvacias

    # Remove zero-labeled pixels for Kappa and F1 score calculation
    non_zero_indices = np.where((output != 0) & (truth != 0))
    output_flat = output[non_zero_indices]
    truth_flat = truth[non_zero_indices]

    kappa = cohen_kappa_score(truth_flat, output_flat)
    f1 = f1_score(truth_flat, output_flat, average="weighted")

    if show:
        print("* Accuracy (pixels)")
        for i in range(1, nclases + 1):
            print(f"    Class {i:02d}: {class_aa[i]:02.02f}")
        print(f"* Accuracy (pixels), OA={OA:02.02f}, AA={AA:02.02f}")
        print(f"    total: {total}, correct: {correct}")

    return (OA, AA, class_aa, kappa, f1)


def print_train_history(
    name,
    losses=None,
    losses_validation=None,
    train_accuracies=None,
    val_accuracies=None,
    show=False,
):
    """
    Plots and saves the training and validation history for loss and accuracy.

    Args:
        name (str): The base name for the output plot files.
        losses (list, optional): List of training losses per epoch. Defaults to None.
        losses_validation (list, optional): List of validation losses per epoch. Defaults to None.
        train_accuracies (list, optional): List of training accuracies per epoch. Defaults to None.
        val_accuracies (list, optional): List of validation accuracies per epoch. Defaults to None.
        show (bool, optional): If True, displays the plot. Otherwise, saves it to a file.
                               Defaults to False.
    """
    fig_size = (10, 5)

    if losses is not None and losses_validation is None:
        plt.figure(figsize=fig_size)
        plt.plot(losses)
        plt.title("Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid()
        if show:
            plt.show()
        else:
            plt.savefig(f"{name}_loss.png")
            plt.close()

    if losses is not None and losses_validation is not None:
        plt.figure(figsize=fig_size)
        plt.plot(losses, label="Train Loss")
        plt.plot(losses_validation, label="Validation Loss")
        plt.title("Train & Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid()
        if show:
            plt.show()
        else:
            plt.savefig(f"{name}_train_val_loss.png")
            plt.close()

    if train_accuracies is not None and val_accuracies is None:
        plt.figure(figsize=fig_size)
        plt.plot(train_accuracies)
        plt.title("Train Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.grid()
        if show:
            plt.show()
        else:
            plt.savefig(f"{name}_train_acc.png")
            plt.close()

    if train_accuracies is not None and val_accuracies is not None:
        plt.figure(figsize=fig_size)
        plt.plot(train_accuracies, label="Train Accuracy")
        plt.plot(val_accuracies, label="Validation Accuracy")
        plt.title("Train & Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid()
        if show:
            plt.show()
        else:
            plt.savefig(f"{name}_train_val_acc.png")
            plt.close()


def print_train_history_subplots(
    name,
    losses=None,
    losses_validation=None,
    train_accuracies=None,
    val_accuracies=None,
    show=False,
):
    """
    Plots and saves the training and validation history for loss and accuracy in a single
    figure with two subplots.

    Args:
        name (str): The base name for the output plot file.
        losses (list, optional): List of training losses per epoch. Defaults to None.
        losses_validation (list, optional): List of validation losses per epoch. Defaults to None.
        train_accuracies (list, optional): List of training accuracies per epoch. Defaults to None.
        val_accuracies (list, optional): List of validation accuracies per epoch. Defaults to None.
        show (bool, optional): If True, displays the plot. Otherwise, saves it to a file.
                               Defaults to False.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    if losses is not None and losses_validation is not None:
        ax1.plot(losses, label="Train Loss")
        ax1.plot(losses_validation, label="Validation Loss")
        ax1.set_title("Train & Validation Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.grid()

    if train_accuracies is not None and val_accuracies is not None:
        ax2.plot(train_accuracies, label="Train Accuracy")
        ax2.plot(val_accuracies, label="Validation Accuracy")
        ax2.set_title("Train & Validation Accuracy")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.legend()
        ax2.grid()

    if show:
        plt.show()
    else:
        plt.savefig(f"{name}_train_val_history.png")
        plt.close()
