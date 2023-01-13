
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import os
import pandas as pd
import numpy as np


class CoronaTweets(Dataset):
    def __init__(self, reviews, targets, tokenizer, max_len=512, transform=None):
        self.reviews = reviews
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.transform = transform
        
        

        assert len(self.tweets) == len(
            self.labels
        ), "Number of tweets does not match the number of labels"

    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.tweets[idx],
            self.labels[idx],
        )

            
data_tweets = CoronaTweets("test")
print(data_tweets[5])




    