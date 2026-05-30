import torchvision
import torch
from torchvision import datasets, transforms


def load_data(root_dir, domain, batch_size):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image_folder = datasets.ImageFolder(
        root=root_dir + domain,
        transform=transform
    )
    data_loader = torch.utils.data.DataLoader(
        dataset=image_folder,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True
    )
    return data_loader


def load_test(root_dir, domain, batch_size):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image_folder = datasets.ImageFolder(
        root=root_dir + domain,
        transform=transform
    )
    data_loader = torch.utils.data.DataLoader(
        dataset=image_folder,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )
    return data_loader