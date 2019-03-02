import torch
from torch.utils import data
from torchvision import datasets, transforms


def get_image_datasets(data_dir):
    # transformations to be applied to all images
    transform = transforms.Compose([
        transforms.ToTensor(),                                   # transform to tensor of values in range [0., 1.]
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # download CIFAR10 train and validation datasets
    image_datasets = {
        'train':
        datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform),
        'validation':
        datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
    }
    return image_datasets


def get_dataloaders(data_dir, batch_size=128):
    image_datasets = get_image_datasets(data_dir)
    dataloaders = {
        'train':
            torch.utils.data.DataLoader(image_datasets['train'],
                                        batch_size=batch_size,
                                        shuffle=True, num_workers=4),
        'validation':
            torch.utils.data.DataLoader(image_datasets['validation'],
                                        batch_size=batch_size,
                                        shuffle=False, num_workers=4)
    }
    return dataloaders
