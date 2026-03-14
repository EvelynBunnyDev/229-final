import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from os.path import join
import numpy as np
import os 

# dataset that gets the first frame from video -> to construct an image dataset
class ImageDataset(Dataset):
    def __init__(self, paths, labels, directory, transform=None, extension='mp4'):
        self.video_dir = directory
        self.image_paths = paths
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.transform = transform if transform else transforms.ToTensor()
        self.extension = extension

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        cap = cv2.VideoCapture(join(self.video_dir, self.image_paths[idx]+'.'+self.extension))
        _, image = cap.read()
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        image = self.transform(image)
        cap.release()
        label = self.labels[idx]
        return image, label

class VideoDataset(Dataset):
    def __init__(self, paths, labels, directory, transform=None, num_frames = 5, extension='mp4'):
        self.video_dir = directory
        self.image_paths = paths
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.transform = transform if transform else transforms.ToTensor()
        self.num_frames = num_frames
        self.extension = extension

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        cap = cv2.VideoCapture(join(self.video_dir, self.image_paths[idx]+'.'+self.extension))
        frames = []

        for i in range(self.num_frames):
            _, frame = cap.read()
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if self.transform:
                image = self.transform(image)

            frames.append(image)

        cap.release()

        video = torch.permute(torch.stack(frames), (1, 0, 2, 3))
        label = self.labels[idx]
        return video, label