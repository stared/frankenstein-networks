# now as it is, from (IN-16)

import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils import data
import torchvision
from torchvision import datasets, transforms
import torchvision.models as models

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

## Custom modules

# antipattern in PyTorch, don't do it!
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

def make_connector(in_channels, out_channels, intermediate_channels=None):
    if not intermediate_channels:
        intermediate_channels = in_channels

    return nn.Sequential(

        nn.Conv2d(in_channels, intermediate_channels, kernel_size=1, padding=0),
        nn.ReLU(inplace=True),
        nn.Conv2d(intermediate_channels, out_channels, kernel_size=1, padding=0),
        nn.ReLU(inplace=True)
    )

class SewnConvNet(nn.Module):
    def __init__(self, net_before, net_after, connector):
        super().__init__()
        self.net_before = net_before.eval()
        self.net_after = net_after.eval()
        self.connector = connector

        self._assert_channels()

    def forward(self, x):
        x = self.net_before(x)
        x = self.connector(x)
        x = self.net_after(x)
        return x

    def _assert_channels(self):
        '''
        net_before, net_after, connector - nn.Sequential or nn.ModuleList
        '''
        before_out_channels = self._find_out_channels(self.net_before)
        connector_in_channels = self._find_in_channels(self.connector)
        if before_out_channels != connector_in_channels:
            raise ValueError('Connector has {} input channels, expected {}.'.format(
                connector_in_channels, before_out_channels)
            )

        connector_out_channels = self._find_out_channels(self.connector)
        after_in_channels = self._find_in_channels(self.net_after)
        if after_in_channels != connector_out_channels:
            raise ValueError('Connector has {} output channels, expected {}.'.format(
                connector_out_channels, after_in_channels)
            )

    def parameters(self, recurse=True):
        return self.connector.parameters()

    def train(self, mode=True):
        self.training = mode
        self.connector.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def _find_in_channels(self, half):
        for m in half.modules():
            if isinstance(m, nn.Conv2d):
                return m.in_channels
            if isinstance(m, nn.Linear):
                return m.in_features

    def _find_out_channels(self, half):
        out_channels = None
        for m in half.modules():
            if isinstance(m, nn.Conv2d):
                out_channels = m.out_channels
            if isinstance(m, nn.Linear):
                out_channels = m.out_features
        return out_channels

def freeze_weights(module):
    for param in module.parameters():
            param.requires_grad = False

## Get some data

from PIL import Image
from pathlib import Path
from torch.utils import data

class ImgNetDataset(data.Dataset):
    def __init__(self, root_dir, transform=None, train=True):
        self.root_dir = Path(root_dir)
        self.transform = transform

        with (self.root_dir / 'ILSVRC2012_validation_ground_truth.txt').open() as f:
            self.labels = [int(line.strip()) - 1 for line in f.readlines()]

        self.images = list((self.root_dir / 'ILSVRC2012_img_val/').glob('*.JPEG'))

        split_idx = 8 * len(self.labels) // 10
        if train:
            self.labels = self.labels[:split_idx]
            self.images = self.images[:split_idx]
        else:
            self.labels = self.labels[split_idx:]
            self.images = self.images[split_idx:]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        img_path = str(self.images[idx].resolve())
        image = Image.open(img_path)
        image = image.convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label

## training ans stuff

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

data_transforms = {
    'train':
    transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]),
    'validation':
    transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        normalize
    ]),
}

image_datasets = {
    'train': ImgNetDataset('/input/',transform=data_transforms['train'], train=True),
    'validation': ImgNetDataset('/input/',transform=data_transforms['validation'], train=False)
}

dataloaders = {
    'train':
    torch.utils.data.DataLoader(image_datasets['train'],
                                batch_size=32,
                                shuffle=True, num_workers=4),
    'validation':
    torch.utils.data.DataLoader(image_datasets['validation'],
                                batch_size=32,
                                shuffle=False, num_workers=4)
}

## training functions

import neptune

ctx = neptune.Context()

def neptune_log_scalars(epoch, logs={}):
    # logging numeric channels
    ctx.channel_send('Accuracy training', epoch, logs['accuracy'])
    ctx.channel_send('Accuracy validation', epoch, logs['val_accuracy'])
    ctx.channel_send('Log-loss training', epoch, logs['log loss'])
    ctx.channel_send('Log-loss validation', epoch, logs['val_log loss'])

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def train_model(model, criterion, optimizer, num_epochs=10):
    model = model.to(device)

    for epoch in range(num_epochs):
        logs = {}
        for phase in ['train', 'validation']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for idx, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                if idx % 100 == 0:
                    print('Epoch [{}/{}]: {} [{}/{}] loss: {}'.format(
                        epoch, num_epochs,
                        phase,
                        idx, len(dataloaders[phase]),
                        loss.item()
                    ))

                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += (preds == labels.data).sum().item()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects / len(dataloaders[phase].dataset)

            prefix = ''
            if phase == 'validation':
                prefix = 'val_'

            logs[prefix + 'log loss'] = epoch_loss
            logs[prefix + 'accuracy'] = epoch_acc

        neptune_log_scalars(epoch, logs)
        # TODO: log false predictions
    return model

## VGG

class Vgg16Limbs(nn.Module):

    def __init__(self, layer_num, left=True):
        super(Vgg16Limbs, self).__init__()
        vgg16 = models.vgg16(pretrained=True)
        freeze_weights(vgg16)
        vgg16_modules = list(vgg16.children())

        if left:
            self.model = nn.Sequential(
                *vgg16_modules[0][:layer_num]
            )
        else:
            self.model = nn.Sequential(
                *vgg16_modules[0][layer_num:],
                Flatten(),
                *vgg16_modules[1]
            )

    def forward(self, x):
        x = self.model(x)
        return x

## run!

vgg_left = Vgg16Limbs(layer_num=12, left=True)
vgg_right = Vgg16Limbs(layer_num=12, left=False)
connector1 = make_connector(256, 256)

sewn_model1 = SewnConvNet(vgg_left, vgg_right, connector1)
criterion = nn.CrossEntropyLoss()
optimizer1 = optim.Adam(sewn_model1.parameters(), lr=1e-2)

sewn_model1_trained = train_model(sewn_model1, criterion, optimizer1, num_epochs=20)
