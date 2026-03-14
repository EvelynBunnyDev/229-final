import torchvision.transforms as T
import torch 
import pickle
import numpy as np
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, random_split

from dataset import ImageDataset, VideoDataset

from os.path import join

def center_crop(img):
    return v2.functional.crop(img, top=115, left=260, height=150, width=150)

def get_transform(model_name, name, stimulus_size):
    if model_name == 'r3d18':
        mean = [0.43216, 0.394666, 0.37645]
        std=[0.22803, 0.22145, 0.216989]
    elif model_name == 'resnet18':
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]
    elif model_name == 'dorsalnet':
        mean = [0.482, 0.482, 0.482]
        std = [0.294, 0.294, 0.294]
    else:
        # default transforms
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]

    if 'dorsal' in name.lower(): 
        transform = T.Compose([T.Lambda(center_crop),  # crop every video to 360 x 360
                    T.Resize(stimulus_size),
                    T.ToTensor(),
                    T.Normalize(mean=mean,
                                std=std)])
    elif 'ventral' in name.lower():
        transform = T.Compose([T.Resize(stimulus_size),
                                T.ToTensor(),
                                T.Normalize(mean=mean,
                                            std=std)]) 
    else:
        raise ValueError(f"Name {name} not supported for transforms.")   

    return transform 
                                    
def get_stimulus_datasets(name, modality='image', stimulus_size=224, model_name=None, transform=None):
    with open(f'../dataset/{name}_dataset.pickle', 'rb') as f:
        data = pickle.load(f)
    
    directory = join("../dataset/", name)
    
    if transform is None:
        transform = get_transform(model_name, name, stimulus_size)

    if modality == 'image':
        train_dataset = ImageDataset(data['train_stimuli'], data['train_activity'], directory, transform=transform)
        test_dataset = ImageDataset(data['test_stimuli'], data['test_activity'], directory, transform=transform)
    elif modality == 'video':
        train_dataset = VideoDataset(data['train_stimuli'], data['train_activity'], directory, transform=transform)
        test_dataset = VideoDataset(data['test_stimuli'], data['test_activity'], directory, transform=transform)
    else:
        raise ValueError(f"Modality {modality} not supported")

    return train_dataset, test_dataset

def get_dataloaders(train_dataset, test_dataset, batch_size=32, val_prop=0.1, seed=42, num_workers=8):
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(train_dataset, [1-val_prop, val_prop], generator=generator)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader, test_loader