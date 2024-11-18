#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# run training

"""
Usage:
    run.py --function=<function> --variant=<attention-model> --pretrain_corpus_path=<file> [--writing_params_path=<file>] [--finetune_corpus_path=<file>] [--reading_params_path=<file>] [--eval_corpus_path=<file>] [--outputs_path=<file>] [options]

Options:
    -h --help                               show this screen.
    --compile                               compile the model
    --no-compile                            do not compile the model
    --backend=<str>                         backend to be used for compilation [default: inductor] {inductor,aot_eager,cudagraphs}
    --function=<function>                   Whether to 'pretrain', 'finetune' or 'evaluate' a model
    --variant=<attention-model>             Which variant of the model to run ('vanilla', 'perceiver')
    --pretrain_corpus_path=<file>           Path of the corpus to pretrain on
    --writing_params_path=<file>            Path to save the model after pretraining/finetuning
    --reading_params_path=<file>            If specified, path of the model to load before finetuning/evaluation
    --finetune_corpus_path=<file>           Path of the corpus to finetune on
    --eval_corpus_path=<file>               Path of the corpus to evaluate on
    --outputs_path=<file>                   File to output predictions
    --tb_expt_name=<str>                    debug string for tb log [default: run]
    --bottleneck_dim=<n>                    bottleneck dim [default: 32]
    --pretrain_lr=<value>                   pretraining lr [default: 6e-3]
    --finetune_lr=<value>                   finetuning lr [default: 6e-4]
"""
from docopt import docopt
from datetime import datetime
import torch
import numpy as np
import sys
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from submission import (
    GPT, GPTConfig, TrainerConfig, Trainer, 
    evaluate_places, sample, initialize_vanilla_model, initialize_perceiver_model,
    finetune, pretrain, train
)

from submission.dataset import WikiDataset  # Import your WikiDataset

def create_model(args, mconf):
    if args['--variant'] == 'vanilla':
        return initialize_vanilla_model(mconf)
    elif args['--variant'] == 'perceiver':
        bottleneck_dim = int(args["--bottleneck_dim"])
        return initialize_perceiver_model(mconf, bottleneck_dim)
    else:
        print("Invalid --variant")
        assert False

def evaluate(args, pretrain_dataset, device, model):
    assert args['--outputs_path'] is not None
    assert args['--reading_params_path'] is not None
    assert args['--eval_corpus_path'] is not None

    print(f"Evaluating using file: {args['--eval_corpus_path']}")

    model.load_state_dict(torch.load(args['--reading_params_path'], map_location=device))
    correct = 0
    total = 0
    with open(args['--outputs_path'], 'w', encoding='utf-8') as fout:
        predictions = []
        for line in tqdm(open(args['--eval_corpus_path'], encoding='utf-8')):
            x = line.strip().split()  # Tokenize input as words
            x = x + [pretrain_dataset.MASK_TOKEN]  # Add mask token
            x = torch.tensor([pretrain_dataset.stoi[s] for s in x], dtype=torch.long)[None, ...].to(device)
            pred = sample(model, x, 32, sample=False)[0]
            completion = ' '.join([pretrain_dataset.itos[int(i)] for i in pred])
            pred = completion.split(pretrain_dataset.MASK_TOKEN)[1] if pretrain_dataset.MASK_TOKEN in completion else ""
            predictions.append(pred)
            fout.write(pred + '\n')
        total, correct = evaluate_places(args['--eval_corpus_path'], predictions)

    if total > 0:
        print('Correct: {} out of {}: {}%'.format(correct, total, correct / total * 100))
    else:
        print('Predictions written to {}; no targets provided'.format(args['--outputs_path']))

def setup_device():
    """ Setup the device used by PyTorch.
    """
    device = torch.device("cpu")
    
    if torch.cuda.is_available(): 
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")

    return device

def main():
    """ Main func.
    """
    args = docopt(__doc__)

    # Save the device
    device = setup_device()

    # Keep the block size 128
    block_size = 128
    wiki_dataset = WikiDataset(args['--pretrain_corpus_path'], block_size)

    # We don't suggest you change these hyperparameters, as they're known to work.
    mconf = GPTConfig(wiki_dataset.vocab_size, wiki_dataset.block_size,
        n_layer=4, n_head=8, n_embd=256)

    # Create model
    attention_model = create_model(args, mconf)

    datetime_str = datetime.now().strftime("%Y%m%d-%H%M%S")

    # TensorBoard training log
    writer = SummaryWriter(log_dir='expt/%s/%s_%s_%d_pt_lr_%f_ft_lr_%f_%s' % (
        args['--function'],
        args['--tb_expt_name'],
        args['--variant'],
        int(args['--bottleneck_dim']),
        float(args['--pretrain_lr']),
        float(args['--finetune_lr']),
        datetime_str))

    if args["--compile"]:
        try:
            attention_model = torch.compile(attention_model, backend=args["--backend"])
            print(f"Attention based model compiled")
        except Exception as err:
            print(f"Model compile not supported: {err}")

    attention_model = attention_model.to(device)

    if args['--function'] == "finetune":
        #TODO: Create new function to handle trainer initialization
        assert args['--finetune_corpus_path'] is not None
        assert args['--writing_params_path'] is not None
        reading_params_path, finetune_corpus_path, finetune_lr = args['--reading_params_path'], args['--finetune_corpus_path'], float(args['--finetune_lr'])
        _, trainer_obj = finetune(reading_params_path, finetune_corpus_path, wiki_dataset, block_size, attention_model, finetune_lr, writer)
        train(attention_model, args['--writing_params_path'], trainer_obj)
    elif args['--function'] == "pretrain":
        assert args['--pretrain_corpus_path'] is not None
        assert args['--writing_params_path'] is not None
        pretrain_lr = float(args['--pretrain_lr'])
        _, trainer_obj = pretrain(wiki_dataset, block_size, attention_model, pretrain_lr, writer)
        train(attention_model, args['--writing_params_path'], trainer_obj)
    else:
        evaluate(args, wiki_dataset, device, attention_model)
    
if __name__ == '__main__':
    main()
