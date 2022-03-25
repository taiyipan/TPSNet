import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from basic_block import BasicBlock
from resnet import ResNet
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.sampler import SubsetRandomSampler
from time import time
import traceback
from functools import partial
import os
from filelock import FileLock
from torch.utils.data import random_split
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler

def load_data(data_dir = '/scratch/tp2231/pytorch/hyperparam_tuning_ray_tune/data'):
    '''
    Load CIFAR10 dataset into train and test
    '''
    # define transform
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # get training and test sets
    with FileLock(os.path.expanduser("~/.data.lock")):
        train_data = torchvision.datasets.CIFAR10(
            root = data_dir,
            train = True,
            download = True,
            transform = transform_train
        )
        test_data = torchvision.datasets.CIFAR10(
            root = data_dir,
            train = False,
            download = True,
            transform = transform_test
        )

    return train_data, test_data

# calculate block count per residual layer
def block_count(depth: int) -> int:
    '''
    Verify and compute block count given depth value (total number of convolutional layers in resnet)
    '''
    assert (depth - 4) % 6 == 0
    return (depth - 4) // 6

def train_cifar(config, num_workers = 48, valid_size = 0.1):
    '''
    Training loop for resnet. This function is passed to ray tune run() method.
    '''
    model = ResNet(BasicBlock, config = config)

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
        print('cuda')
        if torch.cuda.device_count() > 1:
            print('cuda: {}'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr = config['lr'], momentum = 0.9, weight_decay = 5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 300)

    train_data, _ = load_data()

    # split training data
    num_train = len(train_data)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    # define samplers
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    # define data loaders
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size = config['batch_size'],
        sampler = train_sampler,
        num_workers = num_workers
    )
    valid_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size = config['batch_size'],
        sampler = valid_sampler,
        num_workers = num_workers
    )

    for epoch in range(200):  # loop over the dataset multiple times
        running_loss = 0.0
        epoch_steps = 0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            epoch_steps += 1
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / epoch_steps))
                running_loss = 0.0

        # Validation loss
        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0
        for i, data in enumerate(valid_loader, 0):
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loss = criterion(outputs, labels)
                val_loss += loss.cpu().numpy()
                val_steps += 1

        # update scheduler
        scheduler.step()

        tune.report(loss=(val_loss / val_steps), accuracy=correct / total)
    print("Finished Training")

def main(num_samples = 2, max_num_epochs = 1, cpus_per_trial = 48, gpus_per_trial = 4):
    '''
    Main: num_samples determines how many trial models we run, and max_num_epochs determines maximum epochs for each trial.
    '''
    start = time()
    # configure search space
    config = {
        'n': tune.choice([block_count(x) for x in range(16, 83, 6)]),
        'k': tune.choice([1, 2]),
        'lr': tune.loguniform(1e-4, 1e-1),
        'batch_size': tune.choice([32, 64, 128, 256]),
        'net_p': tune.uniform(0.0, 0.5),
        'block_p': tune.uniform(0.0, 0.5)
    }
    scheduler = ASHAScheduler(
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)
    result = tune.run(
        tune.with_parameters(train_cifar),
        resources_per_trial = {"cpu": cpus_per_trial, "gpu": gpus_per_trial},
        config = config,
        metric = "loss",
        mode = "min",
        num_samples = num_samples,
        scheduler = scheduler,
        verbose = 3
    )

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))

    print('Time elapsed: {} hours'.format((time() - start) / 3600.0))

main()
