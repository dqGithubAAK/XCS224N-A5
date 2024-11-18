from .model import GPT
from .dataset import NameDataset
from .trainer import Trainer, TrainerConfig


from .model import GPT
from .dataset import NameDataset
from .trainer import Trainer, TrainerConfig

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
    trainer_obj = None #Trainer object (see trainer.py for more details)
    tconf = None #TrainerConfig object (see trainer.py for more details)
    ### START CODE HERE
    if reading_params_path:
        state_dict = torch.load(reading_params_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict, strict=False) 
        tconf = TrainerConfig(
            max_epochs=5,  # Reduced epochs to balance training time vs. performance
            batch_size=128,  # Reduced batch size for better GPU memory management
            learning_rate=finetune_lr,
            lr_decay=True,
            warmup_tokens=512 * 20,
            final_tokens=200 * len(pretrain_dataset) * block_size,
            num_workers=4,  # Increased to speed up data loading
        )
        
        
    else: 
        tconf = TrainerConfig(
            max_epochs=50,  # Reduced epochs to speed up training
            batch_size=128,  # Reduced batch size for GPU memory limitations
            learning_rate=finetune_lr,
            lr_decay=True,
            warmup_tokens=512 * 20,
            final_tokens=200 * len(pretrain_dataset) * block_size,
            num_workers=4,  # Increased to enhance data loading speed
        )
        
    finetuning_data = NameDataset(open(finetune_corpus_path, encoding='utf-8').read(), pretrain_dataset)
    trainer_obj = Trainer(model, finetuning_data, None, tconf)
    ### END CODE HERE
    return tconf, trainer_obj

def pretrain(pretrain_dataset, block_size, model, pretrain_lr=6e-3, writer=None):
    trainer_obj = None #Trainer object (see trainer.py for more details)
    tconf = None #TrainerConfig object (see trainer.py for more details)

    ### START CODE HERE
    tconf = TrainerConfig(
        max_epochs=300,  # Reduced epochs for reasonable training time
        batch_size=64,  # Reduced batch size to manage memory effectively
        learning_rate=pretrain_lr,
        lr_decay=True,
        warmup_tokens=512 * 20,
        final_tokens=200 * len(pretrain_dataset) * block_size,
        num_workers=4,  # Set to 4 for better data loading performance
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
