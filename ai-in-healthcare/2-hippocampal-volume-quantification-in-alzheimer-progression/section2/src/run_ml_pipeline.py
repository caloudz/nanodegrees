"""
This file contains code that will kick off training and testing processes
"""
import os
import json
import argparse

import numpy as np

from torch.utils.data import random_split

from experiments.UNetExperiment import UNetExperiment
from data_prep.HippocampusDatasetLoader import LoadHippocampusData

class Config:
    """
    Holds configuration parameters
    """
    def __init__(self):
        self.name = "Basic_unet"
        self.root_dir = r"data/"
        self.n_epochs = 10
        self.learning_rate = 0.0002
        self.batch_size = 8
        self.patch_size = 64
        self.test_results_dir = "out/results"
        self.model_name = "" # model name via command line to save weights
        self.weights_name = ""  # weights file name via command line to load weights
        
    def set_model_name(self, m):
        self.model_name = m
        
    def set_weights_name(self, w):
        self.weights_name = w

if __name__ == "__main__":
    # Get configuration

    c = Config()

    parser = argparse.ArgumentParser()
    parser.add_argument("--loadweights", "-lw", help="file name for loading model weights from saved model", action="store")
    parser.add_argument("--savemodel", "-sm", help="file name for saving model weights", action="store")
    args = parser.parse_args()
    
    if args.loadweights:
        print("Load model weights from ", args.loadweights)
        c.set_weights_name(args.loadweights)
    else:
        print("Training new model...")
        
    if args.savemodel:
        print("Storing model weights in ", args.savemodel)
        c.set_model_name(args.savemodel)
    
    # Load data
    print("Loading data...")
    data = LoadHippocampusData(c.root_dir, y_shape = c.patch_size, z_shape = c.patch_size)

    # Split data
    len_data = len(data)
    keys = range(len_data)
    
    train_proportion = 0.7
    val_proportion = 0.2
    test_proportion = 0.1
    
    splits = [int(np.floor(train_proportion * len_data)), 
              int(np.floor(val_proportion * len_data)), 
              int(np.floor(test_proportion * len_data))]
    train, val, test = random_split(keys, splits)
    
    split = {"train": train,
             "val": val, 
             "test": test}

    # Set up and run experiment
    exp = UNetExperiment(c, split, data)

    # Free up memory by deleting the dataset as it has been copied into loaders
    del data
    
    # run training
    exp.run()

    # prep and run testing
    results_json = exp.run_test()
    results_json["config"] = vars(c)
    with open(os.path.join(exp.out_dir, "results.json"), 'w') as out_file:
        json.dump(results_json, out_file, indent=2, separators=(',', ': '))