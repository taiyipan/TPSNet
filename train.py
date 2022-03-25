import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torchsummary import summary

from time import time
import traceback
import os
import argparse

from models import ResNet, BasicBlock


# calculate block count per residual layer
def block_count(depth: int) -> int:
    assert (depth - 4) % 6 == 0
    return (depth - 4) // 6

def get_num_blocks(depth: int) -> list:
    return [block_count(depth), block_count(depth), block_count(depth)]

def make_model(k = 2, d = 82):
    # instantiate model
    model = ResNet(BasicBlock, get_num_blocks(d), k = k)
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
        print('cuda')
        if torch.cuda.device_count() > 1:
            print('cuda: {}'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
    model.to(device)

    # load best model (lowest validation loss)
    #try:
    #    model.load_state_dict(torch.load('./top_models/tpsnet.pt'))
    #    print('Model weights loaded')

    #except:
    #    traceback.print_exc()
    return model


if __name__ == "__main__":
    

    parser = argparse.ArgumentParser(description="Deep Learning Project-1")
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from latest checkpoint')
    parser.add_argument('--epochs', '-e', type=int, default=300, help='no. of epochs')
    parser.add_argument('-w','--num_workers',type=int,default=12,help='number of workers')
    parser.add_argument('-b','--batch_size',type=int,default=128,help='batch_size')
    args = parser.parse_args()   

    # hyperparams
    num_workers = args.num_workers
    batch_size = args.batch_size
    n_epochs = args.epochs

    # define transform
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding = 4),
        transforms.RandomRotation(5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # get training and test sets
    train_data = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform_train)

    test_data = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform_test)

    # define classes
    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # define data loaders
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size = batch_size,
        shuffle = True,
        num_workers = num_workers
    )
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size = batch_size,
        shuffle = False,
        num_workers = num_workers
    )


    model = make_model()
    summary(model, (3, 32, 32))

    # define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum = 0.9, weight_decay = 5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = n_epochs)

    # define training loop
    test_loss_min = np.Inf

    train_loss_list = list()
    test_loss_list = list()
    train_acc_list = list()
    test_acc_list = list()

    start = time()
    for epoch in range(1, n_epochs + 1):
        train_loss = 0
        test_loss = 0
        total_correct_train = 0
        total_correct_test = 0
        total_train = 0
        total_test = 0
        # train model
        model.train()
        for data, target in train_loader:
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data.size(0)
            # calculate accuracies
            _, pred = torch.max(output, 1)
            correct_tensor = pred.eq(target.data.view_as(pred))
            correct = np.squeeze(correct_tensor.numpy()) if not torch.cuda.is_available() else np.squeeze(correct_tensor.cpu().numpy())
            total_correct_train += np.sum(correct)
            total_train += correct.shape[0]

        # validate model
        model.eval()
        for data, target in test_loader:
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            with torch.no_grad():
                output = model(data)
                loss = criterion(output, target)
                test_loss += loss.item() * data.size(0)
                # calculate accuracies
                _, pred = torch.max(output, 1)
                correct_tensor = pred.eq(target.data.view_as(pred))
                correct = np.squeeze(correct_tensor.numpy()) if not torch.cuda.is_available() else np.squeeze(correct_tensor.cpu().numpy())
                total_correct_test += np.sum(correct)
                total_test += correct.shape[0]

        # update scheduler
        scheduler.step()

        # compute average loss
        train_loss /= total_train
        test_loss /= total_test

        # compute accuracies
        train_acc = total_correct_train / total_train * 100
        test_acc = total_correct_test / total_test * 100

        # save data
        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

        # display stats
        print('Epoch: {}/{} \tTrain Loss: {:.6f} \tTest Loss: {:.6f} \tTrain Acc: {:.2f}% \tTest Acc: {:.2f}%'.format(epoch, n_epochs, train_loss, test_loss, train_acc, test_acc))

        # save best model
        if test_loss <= test_loss_min:
            print('Test loss decreased ({:.6f} --> {:.6f}. Saving model...'.format(test_loss_min, test_loss))

            if not os.path.isdir('best_model'):
                os.mkdir('best_model')

            torch.save(model.state_dict(), './best_model/tpsnet.pt')
            test_loss_min = test_loss
    end = time()

    print('Time elapsed: {} hours'.format((end - start) / 3600.0))

    model = make_model()

    # test model
    test_loss = 0
    total_correct = 0
    total = 0

    model.eval()
    for data, target in test_loader:
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        with torch.no_grad():
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item() * data.size(0)
            # calculate accuracies
            _, pred = torch.max(output, 1)
            correct_tensor = pred.eq(target.data.view_as(pred))
            correct = np.squeeze(correct_tensor.numpy()) if not torch.cuda.is_available() else np.squeeze(correct_tensor.cpu().numpy())
            total_correct += np.sum(correct)
            total += correct.shape[0]

    # calculate overall accuracy
    print('Model accuracy on test dataset: {:.2f}%'.format(total_correct / total * 100))


    if not os.path.isdir('results'):
        os.mkdir('results')


    # plot and save figures
    plt.figure()
    plt.plot(np.arange(n_epochs), train_loss_list)
    plt.plot(np.arange(n_epochs), test_loss_list)
    plt.title('Learning Curve: Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train Loss', 'Test Loss'])
    plt.savefig('./results/train_test_loss.png')
    plt.close()

    plt.figure()
    plt.plot(np.arange(n_epochs), train_acc_list)
    plt.plot(np.arange(n_epochs), test_acc_list)
    plt.title('Learning Curve: Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train Accuracy', 'Test Accuracy'])
    plt.savefig('./results/train_test_acc.png')
    plt.close()

    # write training data to csv file
    with open('./results/train_data.csv', 'w') as f:
        f.write('train_loss, test_loss, train_acc, test_acc\n')
        for i in range(n_epochs):
            f.write('{}, {}, {}, {}\n'.format(train_loss_list[i], test_loss_list[i], train_acc_list[i], test_acc_list[i]))
