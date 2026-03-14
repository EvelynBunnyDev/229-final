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
import time

def smoothing_laplacian_loss(data, device, weight=1e-3, L=None):
    if L is None:
        L = torch.tensor([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]],device=device)
        
    temp = torch.reshape(data.squeeze(), [data.squeeze().shape[0],
                          np.sqrt(data.squeeze().shape[1]).astype('int'),
                          np.sqrt(data.squeeze().shape[1]).astype('int')])
    temp = torch.square(F.conv2d(temp.unsqueeze(1),L.unsqueeze(0).unsqueeze(0).float(),
                    padding=1))
    return weight * torch.sum(torch.sqrt(torch.sum(temp, dim=tuple(range(1, temp.ndim))))) # ensure positive 

def train_model(learning_rate, weight_decay, smooth_weight, layer, model_name, lr_decay_step, lr_decay_gamma, num_neurons, num_epochs, train_loader, val_loader, save_path = None, log_path = None, modality='image', stimulus_size=224, endtoend=False):
    pretrained_model = get_pretrained_model(model_name)
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    # init model 
    if modality == 'image':
        net = ImageModel(pretrained_model, layer, num_neurons, device, input_shape=(1, 3, stimulus_size, stimulus_size)).to(device)
    elif modality == 'video':
        net = VideoModel(pretrained_model, layer, num_neurons, device, input_shape=(1, 3, 5, stimulus_size, stimulus_size), endtoend=endtoend).to(device)
    else:
        raise ValueError(f"Modality {modality} not supported")

    # loss function
    criterion = torch.nn.MSELoss()

    # optimizer and lr scheduler
    if endtoend:
        optimizer = torch.optim.Adam(net.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay)        
    else:
        optimizer = torch.optim.Adam(
            [net.w_s, net.w_f],
            lr=learning_rate,
            weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=lr_decay_step,
        gamma=lr_decay_gamma)

    optimizer.step()

    # clear log
    if log_path is not None: 
        if os.path.exists(log_path):
            os.remove(log_path)
            
    for epoch in (range((num_epochs))):
        epoch_loss = 0.0
        epoch_loss_smooth = 0.0
        epoch_loss_recon = 0.0
        for data, label in tqdm(train_loader):
            data = data.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            output = net(data)

            # ignore nans
            nan_idx = torch.isnan(label) | torch.isinf(label)
            output[nan_idx] = 0
            label[nan_idx] = 0

            output = output.squeeze()

            loss_recon = criterion(output, label.float().squeeze())
            loss_smooth = smoothing_laplacian_loss(
                net.w_s, device, weight=smooth_weight
            )
            loss = loss_recon + loss_smooth
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.data.cpu()
            epoch_loss_smooth += loss_smooth.data.cpu()
            epoch_loss_recon += loss_recon.data.cpu()
            
        with torch.no_grad():
            net.eval()
            outputs = []
            labels = []
            val_loss = 0
            val_loss_smooth = 0
            val_loss_recon = 0
            for data, label in val_loader:
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

                loss_recon = criterion(output, label.float().squeeze()) 
                loss_smooth = smoothing_laplacian_loss(
                    net.w_s, device, weight=smooth_weight
                )
                val_loss += loss_recon + loss_smooth
                val_loss_smooth += loss_smooth
                val_loss_recon += loss_recon
                
                outputs.append(output)
                labels.append(label)

            outputs = torch.row_stack(outputs)
            labels = torch.row_stack(labels)
            corrs = []
            for n in range(outputs.shape[1]):
                corrs.append(
                    np.corrcoef(outputs[:, n].cpu().detach().numpy(), labels[:, n].cpu().detach().numpy())[1, 0]
                )
            val_corr = np.nanmean(corrs)
            net.train()
                
        print('======')
        no_epoch = epoch+1 / (num_epochs)
        mean_train_loss = epoch_loss.data.cpu() / len(train_loader)
        mean_train_loss_recon = epoch_loss_recon.data.cpu() / len(train_loader)
        mean_train_loss_smooth = epoch_loss_smooth.data.cpu() / len(train_loader)
        mean_val_loss = val_loss.data.cpu() / len(val_loader)
        mean_val_loss_recon = val_loss_recon.data.cpu() / len(val_loader)
        mean_val_loss_smooth = val_loss_smooth.data.cpu() / len(val_loader)
        
        print("epoch ", epoch, " train loss: ", mean_train_loss)
        print("epoch ", epoch, " val loss: ", mean_val_loss)
        print("epoch ", epoch, " validation corr: ", val_corr)
        print("epoch ", epoch, " learning rate: ", scheduler.get_last_lr())
    
        if log_path is not None: 
            with open(log_path, 'a') as f:
                f.write(
                    'Finished epoch: {} | Train Loss : {:.4f} | Train Loss Smooth : {:.4f} | Train Loss Recon : {:.4f} | '
                    'Val Loss : {:.4f} | Val Loss Smooth : {:.4f} | Val Loss Recon : {:.4f} | Val Corr : {:.4f}\n'.format(
                        epoch + 1,
                        mean_train_loss,
                        mean_train_loss_smooth,
                        mean_train_loss_recon,
                        mean_val_loss,
                        mean_val_loss_smooth,
                        mean_val_loss_recon,
                        val_corr))
        
        scheduler.step()
    
    if save_path is not None:
        torch.save(net.state_dict(), save_path)

    return val_corr

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
    smooth_weight = config.get("smooth_weight", 0.001)
    weight_decay = config.get("weight_decay", 0.0001)
    lr_decay_step = config.get("lr_decay_step", 10)
    lr_decay_gamma = config.get("lr_decay_gamma", 0.1)
    endtoend = config.get("endtoend", False)

    stimulus_sizes = config.get("stimulus_sizes", [112, 224])
    layers = config.get("layers", ["layer1", "layer2", "layer3", "layer4"])
    
    for stimulus_size in stimulus_sizes:
        for layer in layers:
            train_dataset, test_dataset = get_stimulus_datasets(name, modality=modality, stimulus_size=stimulus_size)
        
            train_loader, val_loader, test_loader = get_dataloaders(train_dataset, test_dataset, batch_size=batch_size, val_prop=0.1, seed=42)
        
            save_path = Path(f'./checkpoints/{base_name}_{stimulus_size}_{layer}.pth')
            log_path =  Path(f'./logs/{base_name}_{stimulus_size}_{layer}_train.txt')
        
            n_neurons = train_dataset[0][1].shape[0]
            val_corr = train_model(learning_rate, weight_decay, smooth_weight, layer, model_name, lr_decay_step, lr_decay_gamma, n_neurons, num_epochs, train_loader, val_loader, save_path = save_path, log_path=log_path, modality=modality, stimulus_size=stimulus_size, endtoend=endtoend)

if __name__ == "__main__":
    main()