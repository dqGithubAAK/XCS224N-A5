import random
import torch
from torch.utils.data import Dataset, DataLoader
import argparse

class WikiDataset(Dataset):
    def __init__(self, file_path, block_size):
        # Define special tokens for masking and padding
        self.MASK_TOKEN = "[MASK]"  # Special token for masking
        self.PAD_TOKEN = "[PAD]"    # Special token for padding

        # Load and tokenize data at the word level
        with open(file_path, 'r', encoding='utf-8') as f:
            data = f.read()
        tokenized_data = data.split()  # Split the data into words

        # Create the vocabulary, ensuring special tokens are included
        unique_tokens = list(set(tokenized_data))
        assert self.MASK_TOKEN not in unique_tokens
        assert self.PAD_TOKEN not in unique_tokens
        unique_tokens.insert(0, self.MASK_TOKEN)
        unique_tokens.insert(0, self.PAD_TOKEN)

        # Create mapping from tokens to indices and vice versa
        self.stoi = {tok: i for i, tok in enumerate(unique_tokens)}
        self.itos = {i: tok for i, tok in enumerate(unique_tokens)}

        # Set dataset parameters
        self.vocab_size = len(unique_tokens)
        self.block_size = block_size
        self.data = tokenized_data
        print(f"Data has {len(tokenized_data)} tokens, {self.vocab_size} unique.")

    def __len__(self):
        # Calculate the length of the dataset in terms of blocks
        return len(self.data) // self.block_size

    def __getitem__(self, idx):
        # Get a block of data for training
        start_idx = idx * self.block_size
        end_idx = start_idx + self.block_size
        block = self.data[start_idx:end_idx]

        # Add padding if the block is shorter than block_size
        if len(block) < self.block_size:
            block += [self.PAD_TOKEN] * (self.block_size - len(block))

        # Convert the block to indices
        x = torch.tensor([self.stoi[token] for token in block[:-1]], dtype=torch.long)
        y = torch.tensor([self.stoi[token] for token in block[1:]], dtype=torch.long)

        return x, y

"""
Code under here is strictly for your debugging purposes; feel free to modify
as desired.
"""
if __name__ == '__main__':
    argp = argparse.ArgumentParser()
    argp.add_argument('dataset_type', help="Type of dataset to sample from. Options: wikidata.",
                      choices=["wikidata"])
    argp.add_argument('--file_path', help="Path to the dataset file. Default is './../data/train_word.txt'.",
                      default='./../data/train_word.txt')
    argp.add_argument('--block_size', help="Block size for training data. Default is 128.", type=int,
                      default=128)
    argp.add_argument('--sample_count', help="Number of samples to print. Default is 4.", type=int,
                      default=4)
    args = argp.parse_args()

    if args.dataset_type == 'wikidata':
        # Create the WikiDataset instance with provided file path and block size
        wiki_dataset = WikiDataset(args.file_path, args.block_size)

        # DataLoader for batching (optional for testing)
        data_loader = DataLoader(wiki_dataset, batch_size=1, shuffle=True)

        # Print some samples from the dataset
        for i, (x, y) in zip(range(args.sample_count), data_loader):
            x_tokens = [wiki_dataset.itos[int(idx)] for idx in x[0]]
            y_tokens = [wiki_dataset.itos[int(idx)] for idx in y[0]]
            print(f'Sample {i + 1}:')
            print('x:', ' '.join(x_tokens))
            print('y:', ' '.join(y_tokens))
    else:
        raise ValueError("Unknown dataset type in command line args: {}".format(args.dataset_type))
