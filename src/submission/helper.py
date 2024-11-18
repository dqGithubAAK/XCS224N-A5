from .model import GPT
from .dataset import NameDataset
from .trainer import Trainer, TrainerConfig


"""

##### Remove #####

import sys
from pathlib import Path
import os
import socket

def set_project_root(directory):
    hostname = socket.gethostname()
    current_path = Path.cwd()

    if "MacBook-Pro-4.fritz.box" or "MacBook-Pro-4.local"  in hostname:
        project_root = Path(directory)
    else:
        project_root = None
        for parent in current_path.parents:
            if (parent / 'venv').exists():  # Assuming 'venv' indicates the project root
                project_root = parent
                break
        if project_root is None:
            project_root = current_path

    project_root_str = str(project_root)
    
    # Add the root directory to sys.path
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)

    # Use the current working directory in interactive environments
    script_directory = str(current_path)
    if script_directory not in sys.path:
        sys.path.insert(0, script_directory)
    
    print(f"Project root set to: {project_root_str}")
    print(f"Script directory set to: {script_directory}")


set_project_root('/Users/alexanderkatz/Desktop/ML_NLP/Study_Courses/Stanford_AI/DL_NLP/Assignment_5/XCS224N-A5/src/submission/')
from trainer import Trainer, TrainerConfig
from model import GPT
from dataset import NameDataset

#####

"""


import torch
import random
random.seed(0)

def initialize_vanilla_model(mconf):
    attention_model = None
    ### TODO:
    ### [part c]: Make some model here
    
    ### START CODE HERE
    attention_model = GPT(mconf)
    ### END CODE HERE
    return attention_model

def initialize_perceiver_model(mconf, bottleneck_dim=32):
    attention_model = None
    ### TODO
    ### [part g]: Make some other model here

    ### START CODE HERE
    mconf.bottleneck_dim=bottleneck_dim # model.py for which parameter of config to set True
    mconf.perceiver = True
    attention_model = GPT(mconf)
    ### END CODE HERE
    return attention_model

def finetune(reading_params_path, finetune_corpus_path, pretrain_dataset, block_size, model, finetune_lr=6e-4, writer=None):
    ### TODO:
    ### [part c] [part f]:
    ### - Given:
    ###     1. A finetuning corpus specified in finetune_corpus_path
    ###     2. A path reading_params_path containing pretrained model
    ###         parameters, or None if finetuning without a pretrained model
    ### - Goals:
    ###     1. If reading_params_path is specified, load these parameters
    ###         into the model
    ###     2. Finetune the model on this corpus
    ###
    ### - Make sure to use the following hyperparameters:
    ###     Hyperparameters for finetuning WITHOUT a pretrained model:
    ###         max_epochs=75
    ###         batch_size=256
    ###         learning_rate=6e-4
    ###         lr_decay=True
    ###         warmup_tokens=512*20
    ###         final_tokens=200*len(pretrain_dataset)*block_size
    ###         num_workers=0
    ###     Hyperparameters for finetuning WITH a pretrained model:
    ###         max_epochs=10
    ###         batch_size=256
    ###         learning_rate=6e-4
    ###         lr_decay=True
    ###         warmup_tokens=512*20
    ###         final_tokens=200*len(pretrain_dataset)*block_size
    ###         num_workers=0
    ###
    ###
    ### Note: Please use torch.load(reading_params_path, map_location=torch.device('cpu'), weights_only=True) to load pretrained model 

    trainer_obj = None #Trainer object (see trainer.py for more details)
    tconf = None #TrainerConfig object (see trainer.py for more details)
    ### START CODE HERE
    if reading_params_path:
        state_dict = torch.load(reading_params_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict, strict=False) 
        tconf = TrainerConfig(
            max_epochs=10,
            batch_size=256,
            learning_rate=finetune_lr,
            lr_decay=True,
            warmup_tokens=512 * 20,
            final_tokens=200 * len(pretrain_dataset) * block_size,
            num_workers=0, 
        )
        
        
    else: 
        tconf = TrainerConfig(
            max_epochs=75,
            batch_size=256,
            learning_rate=finetune_lr,
            lr_decay=True,
            warmup_tokens=512 * 20,
            final_tokens=200 * len(pretrain_dataset) * block_size,
            num_workers=0, 
        )
        
    finetuning_data = NameDataset(open(finetune_corpus_path, encoding='utf-8').read(), pretrain_dataset)
    trainer_obj = Trainer(model, finetuning_data, None, tconf)
    ### END CODE HERE
    return tconf, trainer_obj

def pretrain(pretrain_dataset, block_size, model, pretrain_lr=6e-3, writer=None):
    ### TODO:
    ### [part f]:
    ### - Given:
    ###     1. A corpus specified in pretrain_dataset
    ### - Goals:
    ###     1. Pretrain the model on this corpus
    ###
    ### - Make sure to use the following hyperparameters for pretraining:
    ###     max_epochs=650
    ###     batch_size=128
    ###     learning_rate=6e-3
    ###     lr_decay=True
    ###     warmup_tokens=512*20
    ###     final_tokens=200*len(pretrain_dataset)*block_size
    ###     num_workers=0

    trainer_obj = None #Trainer object (see trainer.py for more details)
    tconf = None #TrainerConfig object (see trainer.py for more details)

    ### START CODE HERE
    tconf = TrainerConfig(
        max_epochs=650,
        batch_size=128,
        learning_rate=pretrain_lr,
        lr_decay=True,
        warmup_tokens=512 * 20,
        final_tokens=200 * len(pretrain_dataset) * block_size,
        num_workers=0, 
    )
    trainer_obj = Trainer(model, pretrain_dataset, None, tconf)
    ### END CODE HERE
    return tconf, trainer_obj

def train(model, writing_params_path, trainer_obj):
    ### TODO:
    ### - Given:
    ###     An output path writing_params_path for the model parameters
    ### [part c]:
    ###
    ### Note: trainer_obj is of type Trainer (see trainer.py for more details)

    ### START CODE HERE
    trainer_obj.model = model
    trainer_obj.config.ckpt_path = writing_params_path
    trainer_obj.train()
    ### END CODE HERE
    return
