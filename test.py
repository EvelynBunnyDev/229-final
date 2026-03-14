import torch
import torch.nn as nn
import matplotlib.pyplot as plt 
from dataloader import get_stimulus_datasets, get_dataloaders
from pathlib import Path
from train import train_model
from torch.utils.data import Subset
import numpy as np

from model import get_pretrained_model
from utils import set_seed
from model import ImageModel, VideoModel
import numpy as np
import argparse
import pickle
from utils import load_config

def test_model(load_path, pretrained_model, layer, num_neurons, device, test_loader, modality, stimulus_size, log_path=None, endtoend=False):
    # init model 
    if modality == 'image':
        net = ImageModel(pretrained_model, layer, num_neurons, device, input_shape=(1, 3, stimulus_size, stimulus_size)).to(device)
    elif modality == 'video':
        net = VideoModel(pretrained_model, layer, num_neurons, device, input_shape=(1, 3, 5, stimulus_size, stimulus_size), endtoend=endtoend).to(device)
    else:
        raise ValueError(f"Modality {modality} not supported")

    # load the model
    net.load_state_dict(torch.load(load_path, weights_only=False, map_location=torch.device('cpu')))

    # put model in eval mode so that BN gets evaluated properly 
    net.eval()

    net.to(device)

    # aggregate output and labels for test set 
    outputs = []
    labels = []
    for data, label in test_loader:
        if torch.sum(torch.isnan(data)) == torch.numel(data):
            continue
        data = data.to(device)
        label = label.to(device)
        output = net(data)
        
        # ignore nans
        nan_idx = torch.isnan(label)
        output[nan_idx] = 0
        label[nan_idx] = 0

        output = output.squeeze()

        outputs.append(output)
        labels.append(label)

    outputs = torch.row_stack(outputs)
    labels = torch.row_stack(labels)

    # compute corr to average for each neuron
    corrs = []
    for n in range(outputs.shape[1]):
        corrs.append(
            np.corrcoef(outputs[:, n].cpu().detach().numpy(), labels[:, n].cpu().detach().numpy())[1, 0]
        )

    print(f"=====================")
    print(f"number of neurons: {len(corrs)}")
    print(f"test set mean correlation to average: {np.nanmean(corrs)}")
    print(f"test set stdev correlation to average: {np.nanstd(corrs)}")
    print(f"test set sterr correlation to average: {np.nanstd(corrs)/np.sqrt(len(corrs))}")
    
    if log_path is not None:
        np.save(log_path, np.array(corrs))

    return np.nanmean(corrs), np.nanstd(corrs), np.nanstd(corrs)/np.sqrt(len(corrs))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML config file")
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    name = config["name"]
    base_name = config["encoding_ckpt"]
    
    modality = config.get("modality", "image")
    batch_size = config.get("batch_size", 40)
    model_name = config.get("model_name", "r3d18")
    endtoend = config.get("endtoend", False)
    
    stimulus_sizes = config.get("stimulus_sizes", [112, 224])
    layers = config.get("layers", ["layer1", "layer2", "layer3", "layer4"])
    
    for stimulus_size in stimulus_sizes:
        for layer in layers:
            train_dataset, test_dataset = get_stimulus_datasets(name, modality=modality, stimulus_size=stimulus_size)
        
            train_loader, val_loader, test_loader = get_dataloaders(train_dataset, test_dataset, batch_size=batch_size, val_prop=0.1, seed=42)
        
            num_neurons = train_dataset[0][1].shape[0]
            pretrained_model = get_pretrained_model(model_name)
        
            set_seed()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
            load_path = Path(f'./checkpoints/{base_name}_{stimulus_size}_{layer}.pth')
            log_path =  Path(f'./logs/{base_name}_{stimulus_size}_{layer}_test.npy')
            
            test_model(load_path, pretrained_model, layer, num_neurons, device, test_loader, modality, stimulus_size=stimulus_size, log_path=log_path, endtoend=endtoend)

if __name__ == "__main__":
    main()