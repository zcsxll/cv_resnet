import torch
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt

from model import resnet

def calc_correct(pred, label):
    pred = np.argmax(pred, axis = 1)
    correct = pred == label
    # correct = np.mean(correct.astype(np.float32))
    correct = np.sum(correct)
    return correct

def train(pkl_path=None):
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    #transform = transforms.Compose([transforms.ToTensor(),])
    dataset = CIFAR10('./dataset/data_CIFAR10', train=True, download=False, transform=transform)

    data_loader = torch.utils.data.DataLoader(
        dataset = dataset,
        batch_size = 32,
        shuffle = True,
        num_workers = 32,
        collate_fn = None)

    model = resnet.resnet18(num_classes=10)
    if pkl_path is not None:
        model.load_state_dict(torch.load(pkl_path))
    loss_func = torch.nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr = 0.001)

    total_correct = 0
    for step, (batch_x, batch_y) in enumerate(data_loader):
        # print(step, batch_x.shape, batch_y.shape)
        outputs = model(batch_x)

        total_correct += calc_correct(outputs.cpu().detach().numpy(), batch_y.cpu().detach().numpy())
        acc = total_correct / ((step + 1) * 32)

        # print("====", outputs.shape, batch_y.shape)
        loss = loss_func(outputs, batch_y)
        optim.zero_grad()
        loss.backward()
        optim.step()
        print("%d %.4f %.4f" % (step, loss.cpu().detach().numpy(), acc))
        # plt.figure()
        # plt.imshow(batch_x[0].cpu().detach().numpy().T)
        # plt.savefig("./img.png")

        # if step >= 0:
        #     break

    torch.save(model.state_dict(), "./model.pkl")

if __name__ == "__main__":
    train()
    # train("./model.pkl")