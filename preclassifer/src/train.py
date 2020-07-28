from dataloader import train_loader, FilePaths
from torchvision import models
import torch.nn as nn
import torch
import torch.optim as optim
import timeit
from network import device, optimizer, model, criterion

"""train"""


def train_model_epochs(num_epochs, gpu=True):
    """ Trains the model for a given number of epochs on the training set. """
    for epoch in range(num_epochs):
        length = len(train_loader) // 10

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            images, labels = data

            if gpu:
                images = images.to(device)
                labels = labels.to(device)

            # Zero the parameter gradients means to reset them from
            # any previous values. By default, gradients accumulate!
            optimizer.zero_grad()

            # Passing inputs to the model calls the forward() function of
            # the Module class, and the outputs value contains the return value
            # of forward()
            outputs = model(images)

            # Compute the loss based on the true labels
            loss = criterion(outputs, labels)

            # Backpropagate the error with respect to the loss
            loss.backward()

            # Updates the parameters based on current gradients and update rule;
            # in this case, defined by SGD()
            optimizer.step()

            # Print our loss
            running_loss += loss.item()
            if i % length == length - 1:  # print every 1000 mini-batches
                print('Epoch / Batch [%d / %d] - Loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / length))
                running_loss = 0.0


if __name__ == '__main__':
    if torch.cuda.is_available() is False:
        gpu_train_time = timeit.timeit(
            "train_model_epochs(num_epochs, gpu=False)",
            setup="num_epochs=100",
            number=1,
            globals=globals(),
        )
    else:
        gpu_train_time = timeit.timeit(
            "train_model_epochs(num_epochs, gpu=True)",
            setup="num_epochs=100",
            number=1,
            globals=globals(),
        )
    # 保存模型
    torch.save(model, FilePaths.savedModel)
