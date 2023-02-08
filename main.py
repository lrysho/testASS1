import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
from torchvision.models import vgg16 # TODO use the VGG16 model defined in torchvision

import argparse
import os
import yaml
from easydict import EasyDict

# This is a quite saimple CNN with two convolutional layers
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        ) 
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 7, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 128, 5, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(128 * 6 * 6, 4096), #TODO (input: 128 * width? * height?, output dim)
            nn.ReLU(True),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, num_classes), #TODO (input: 128 * width? * height?, output dim?)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = torch.flatten(x, 1) # TODO
        x = self.classifier(x)
        return x


def train(train_loader, model, loss_fn, optimizer, device):
    for i, (image, annotation) in enumerate(train_loader):
        # move data to the same device as model
        image = image.to(device)
        annotation = annotation.to(device)

        # forward/Compute prediction error
        ousput = model(image)
        loss = loss_fn(ousput, annotation)

        # TODO Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print statistics
        # running_loss += loss.item()
        if i % 20 == 0:    # print every 20 iterates
            print(f'iterate {i + 1}: loss={loss:>7f}')

def val(val_loader, model, device):
    # switch to evaluate mode
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (image, annotation) in enumerate(val_loader):
            # move data to the same device as model
            image = image.to(device)
            annotation = annotation.to(device)

            # forward/Compute prediction error
            output = model(image)

            # for compute accuracy
            _, predicted = torch.max(output.data, 1)
            total += annotation.size(0)
            correct += (predicted == annotation).sum().item()
    # accuracy
    acc = correct / total
    # TODO how to compute accuracy per category
    print(f'val accuracy: {100 * acc:>2f} %')
    return acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    args = parser.parse_args()

    # load config
    with open(args.config, 'r') as f:
        config = EasyDict(yaml.safe_load(f))
    print(config)

    # define image transform
    transform = transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225]),
                    ])
    # loda data
    traindir = os.path.join(config.data, 'train')
    valdir = os.path.join(config.data, 'val')    # load data
    train_dataset = datasets.ImageFolder(traindir,transform)
    val_dataset = datasets.ImageFolder(valdir, transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False)
    
    # device used to train
    device = torch.device("cuda:0")
    
    # load model
    model = SimpleCNN()
    # TODO 
    model.to(device)

    # choose loss function
    loss_fn = nn.CrossEntropyLoss()

    # config optimizer
    optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=0.9, weight_decay=0.0001)
    scheduler = StepLR(optimizer, step_size=config.lr_strategy.step, gamma=config.lr_strategy.gamma) # lr decay 0.1 every epoch

    # create model save path
    os.makedirs(config.work_dir, exist_ok=True)

    max_acc = -float('inf')
    for epoch in range(config.epochs):
        print('-' * 30, 'epoch', epoch + 1, '-' * 30)

        # train
        train(train_loader, model, loss_fn, optimizer, device)
        print('lr: {}'.format(optimizer.param_groups[0]['lr']))

        # validation
        acc = val(val_loader, model, device)

        # save best model
        if acc > max_acc:
            pt_path = os.path.join(config.work_dir, 'best.pt')
            torch.save(model.state_dict(), pt_path)
            print('save model')

        # decay learning rate
        scheduler.step()
    print('Finished Training')