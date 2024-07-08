import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from loader import customdata
from network import Archi
from sklearn.metrics import mean_squared_error
import numpy as np
import os
import matplotlib.pyplot as plt

import pandas as pd

def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    p = []
    t = []

    for batch_idx, (data1, label) in enumerate(train_loader):
        data1, label = data1.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(data1)
        loss = criterion(output[:, 0], label)
        loss.backward()
        optimizer.step()
        op = output[:, 0]
        t.extend(label.detach().cpu().numpy())
        p.extend(op.detach().cpu().numpy())

        if batch_idx % 1 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data1), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    MSE = mean_squared_error(np.array(t), np.array(p), squared=False)
    print(f"MSE: {MSE:.4f}")

def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    p = []
    t = []
    errors = []

    with torch.no_grad():
        for (data1, target) in test_loader:
            data1, target = data1.to(device), target.to(device)
            output = model(data1)
            test_loss += criterion(output[:, 0], target).item()  # sum up batch loss
            op = output[:, 0]
            t.extend(target.detach().cpu().numpy())
            p.extend(op.detach().cpu().numpy())
            error = abs(op.item() - target.item())
            errors.append((error, data1.cpu().numpy(), target.item(), op.item()))

    MSE = mean_squared_error(np.array(t), np.array(p), squared=False)
    print(f'\nTest set: Average loss: {test_loss / len(test_loader.dataset):.4f}, MSE score: {MSE:.4f}\n')

    actual_vs_predicted = pd.DataFrame({'Actual': t, 'Predicted': p})
    print(actual_vs_predicted)

    global maxv
    if MSE < maxv:
        torch.save(model.state_dict(), f"models/test_model_{MSE:.4f}.pth.tar")
        print("model saved")
        maxv = MSE

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    torch.manual_seed(42)

    mt = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
    ])

    trainset = customdata(root="Data/train/", train=True, transforms=mt)
    testset = customdata(root="Data/test/", train=False, transforms=mt)

    train_loader = DataLoader(trainset, batch_size=4, shuffle=False, drop_last=True)
    test_loader = DataLoader(testset, batch_size=1, shuffle=False, drop_last=False)

    model = Archi().to(device)
    print(model)

    criterion_train = nn.MSELoss().to(device)
    criterion_test = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

    num_epochs = 30
    global maxv
    maxv = 100000000

    for epoch in range(num_epochs):
        train(model, device, train_loader, optimizer, criterion_train, epoch)
        test(model, device, test_loader, criterion_test)
