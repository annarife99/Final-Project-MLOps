
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import os
import pandas as pd
import numpy as np


class CoronaTweets(Dataset):
    def __init__(self, type: str = "test") -> None:
        if type == "train":
            file_tweets = "/data/processed/train.pth"
        elif type == "test":
            file_tweets =  "/data/processed/test.pth"
        else:
            raise Exception(f"Unknown Dataset type: {type}")
        
        self.path=os.getcwd()
        self.file = torch.load(self.path+file_tweets)
        self.tweets = self.file['tweets']
        self.labels = self.file['labels']
        

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




    