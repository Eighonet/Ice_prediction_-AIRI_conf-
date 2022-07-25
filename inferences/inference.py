from os import listdir
import json
import os

import pandas as pd
import numpy as np

import torch

def get_files(folders: list) -> list:
    """
       Get .pt filenames with SIC+GFS data.
         *  folders -- list of paths to train/val/test folders.
    """
    train_files, val_files, test_files = sorted([file for file in listdir(folders[0])])[1346:],\
                                         sorted([file for file in listdir(folders[1])]),\
                                         sorted([file for file in listdir(folders[2])])

    files = train_files + val_files + test_files
    return files

def get_coords(grid) -> tuple:
    """
       Get longitude & latitude of selected dataset patches.
         *  grid -- .pt file with grid & aux data.
    """
    lons, lats = grid["lons"].reshape(-1).tolist(), grid["lats"].reshape(-1).tolist()
    return {"lons":lons, "lats":lats}

def get_model(model_name: str, device: str = "cpu"):
    return torch.load(model_name).to(device)

def unet_inference(model, folders: list, files: list, date: str, grid, device: str = "cpu") -> dict():
    """
       Predict future conditions of SIC regarding the date.
         *  date -- reference data.
    """
    def round_tensor(data: torch.Tensor) -> torch.Tensor:
        output_round = torch.round(data)
        output_round[output_round <= 0] = 0
        return output_round
    
    def preprocess_image(tensor: torch.Tensor, grid) -> torch.Tensor:
        return (torch.nan_to_num(tensor, nan=-10.0) + grid['land']*10)
    
    index = files.index(date +".pt")
    context = files[index-6:index+1]
    pred_dates = files[index+1:index+4]
    
    if date[:4] == "2021":    
        images = [preprocess_image(torch.load(folders[2] + context[i])["jaxa.sic"], grid)\
                                for i in range(0, len(context))]
    else:
        if date[:4] == "2020":
            images = [preprocess_image(torch.load(folders[1] + context[i])["jaxa.sic"], grid)\
                                        for i in range(0, len(context))]
        else:
            images = [preprocess_image(torch.load(folders[0] + context[i])["jaxa.sic"], grid)\
                                        for i in range(0, len(context))]
            
    images_tensor = torch.stack(images).to(device)
    model_output = round_tensor(model(images_tensor[None, :, :, :])[0]).detach().numpy()
    result = {pred_dates[i][:-3]:model_output[i].reshape(-1).tolist() for i in range(len(model_output))}
    return result

import json

def save_results(inference_output:dict, dataset:str, date:str, model_name:str) -> None:
    """
       Save predictions as a .json file.
    """
    try:
        os.mkdir('models_output/')
    except:
        pass
    
    with open('models_output/' + dataset + "_" + date + "_" + model_name +'.json', 'w') as f:
        json.dump(inference_output, f)

def main_inference(dataset: str, date: str, model_name: str) -> dict:
    """
       Main wrapper for inferences.
         *  date -- reference data.
    """
    folder_train, folder_val, folder_test = dataset + '/train/maps/', dataset + '/valid/maps/', dataset + '/test/maps/'
    files = get_files([folder_train, folder_val, folder_test])
    grid = torch.load(dataset + "/train/grid.pt")
    
    model = get_model(model_name)
    
    pred_sic = unet_inference(model, [folder_train, folder_val, folder_test], files, date, grid)
    coords = get_coords(grid)
    
    output = {"coords":coords, "magn":pred_sic}
    save_results(output, dataset, date, model_name)
    
    return output