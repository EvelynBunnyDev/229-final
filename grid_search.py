import argparse 
import pickle
from dataloader import get_dataloaders, get_stimulus_datasets

from pathlib import Path
from torch.utils.data import Subset
import numpy as np

import torch
import torch.nn as nn
from model import get_pretrained_model
from utils import set_seed
from model import ImageModel, VideoModel
import torch.nn.functional as F
from utils import load_config
import os 
from tqdm import tqdm 

from train import train_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML config file")
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    name = config["name"]
    base_name = config["encoding_ckpt"]
    
    modality = config.get("modality", "video")
    batch_size = config.get("batch_size", 64)
    model_name = config.get("model_name", "r3d18")
    num_epochs = config.get("num_epochs", 15)
    learning_rate = config.get("learning_rate", 0.001)
    endtoend = config.get("endtoend", False)

    print("**************", model_name, "**************")

    smooth_weights = [0.00001, 0.0001, 0.001] 
    weight_decays = [0.00001, 0.0001, 0.001]    
        
    lr_decay_step = config.get("lr_decay_step", 10)
    lr_decay_gamma = config.get("lr_decay_gamma", 0.1)

    stimulus_sizes = config.get("stimulus_sizes", [112, 224])
    layers = config.get("layers", ["layer1", "layer2", "layer3", "layer4"])
    
    for stimulus_size in stimulus_sizes:
        for layer in layers:
            
            results = []
            val_corrs = []
            for smooth_weight in smooth_weights:
                for weight_decay in weight_decays:
                    print("###### smooth_weight: ", smooth_weight," ###### weight_decay: ", weight_decay)
                    train_dataset, test_dataset = get_stimulus_datasets(name, modality=modality, stimulus_size=stimulus_size)
                
                    train_loader, val_loader, test_loader = get_dataloaders(train_dataset, test_dataset, batch_size=batch_size, val_prop=0.1, seed=42)
                
                    save_path = None
                    log_path = None
                    n_neurons = train_dataset[0][1].shape[0]
                    val_corr = train_model(learning_rate, weight_decay, smooth_weight, layer, model_name, lr_decay_step, lr_decay_gamma, n_neurons, num_epochs, train_loader, val_loader, save_path = save_path, log_path=log_path, modality=modality, stimulus_size=stimulus_size, endtoend=endtoend)
                    results.append((val_corr, smooth_weight, weight_decay))

            # saving results of grid search
            results = np.array(results)
            out_path =  f'./logs/{base_name}_{stimulus_size}_{layer}_grid.npy'
            np.save(out_path, results)
            
            best_ind = np.argmax(results[:, 0])
            smooth_weight = results[best_ind, 1]
            weight_decay = results[best_ind, 2]

            # refit with best hyperparameters & save checkpoint
            train_dataset, test_dataset = get_stimulus_datasets(name, modality=modality, stimulus_size=stimulus_size)
        
            train_loader, val_loader, test_loader = get_dataloaders(train_dataset, test_dataset, batch_size=batch_size, val_prop=0.1, seed=42)
        
            save_path = Path(f'./checkpoints/{base_name}_{stimulus_size}_{layer}.pth')
            log_path =  Path(f'./logs/{base_name}_{stimulus_size}_{layer}_train.txt')
        
            val_corr = train_model(learning_rate, weight_decay, smooth_weight, layer, model_name, lr_decay_step, lr_decay_gamma, n_neurons, num_epochs, train_loader, val_loader, save_path = save_path, log_path=log_path, modality=modality, stimulus_size=stimulus_size, endtoend=endtoend)

if __name__ == "__main__":
    main()