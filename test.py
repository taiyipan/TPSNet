import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.sampler import SubsetRandomSampler
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
    try:
        model.load_state_dict(torch.load('./top_models/tpsnet.pt'))
        print('Model weights loaded')

    except:
        traceback.print_exc()
    return model



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Deep Learning Project-1")
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from latest checkpoint')
    parser.add_argument('--epochs', '-e', type=int, default=200, help='no. of epochs')
    parser.add_argument('-w','--num_workers',type=int,default=12,help='number of workers')
    parser.add_argument('-b','--batch_size',type=int,default=128,help='batch_size')
    args = parser.parse_args()   

    # hyperparams
    num_workers = args.num_workers
    batch_size = args.batch_size
    n_epochs = args.epochs


    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    test_data = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform_test)

   
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size = batch_size,
        num_workers = num_workers
    )

    

    model = make_model()
    summary(model, (3, 32, 32)) 

    # define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # load best model (lowest validation loss)
    try:
        model.load_state_dict(torch.load('./top_models/tpsnet.pt'))
        print('Model weights loaded')
    except:
        traceback.print_exc()

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
    print('total:', total)
    print('total correct:', total_correct)
    # calculate overall accuracy
    print('Model accuracy on test dataset: {:.2f}%'.format(total_correct / total * 100))