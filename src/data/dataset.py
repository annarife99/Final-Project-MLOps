
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import os
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from src.data.clean_functions import preprocessText
#from clean_functions import preprocessText
import torch


class CoronaTweets(Dataset):
    def __init__(self, reviews, targets, tokenizer, max_len=512, transform=None):
        self.reviews = reviews
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.transform = transform
    
    # how large the datset is
    def __len__(self):
        return len(self.reviews)
    
    def __getitem__(self,itemInd):
        reviews =  str(self.reviews[itemInd])

        # implement in transform too
        reviews = preprocessText(reviews)

        if self.transform:
            reviews = self.transform(self.reviews)
    
        encoding = self.tokenizer.encode_plus(
            reviews,
            max_length=self.max_len,
            add_special_tokens=True,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors="pt"
        )
        # encoding["input_ids"] > 1*150 >> .flatten() > 150, > in bach size makes problem
        return {
            'review_text': reviews,
            'input_ids': encoding["input_ids"].flatten(),
            'attention_mask': encoding["attention_mask"].flatten(),
            'targets': torch.tensor(self.targets[itemInd], dtype=torch.long)
        }
        

def create_dataloader(df, tokenizer, max_len, batch_size):
    ds = CoronaTweets(
        reviews = df["Reviews"].to_numpy(),
        targets = df["Sentiment"].to_numpy(),
        tokenizer = tokenizer,
        max_len = max_len
    )

    return DataLoader (
        ds,
        batch_size = batch_size,
        num_workers = 0
    )
            




    