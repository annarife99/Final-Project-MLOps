
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
            file_tweets = "/data/processed/Corona_NLP_train.csv"
        elif type == "test":
            file_tweets =  "/data/processed/Corona_NLP_test.csv"
        else:
            raise Exception(f"Unknown Dataset type: {type}")
        
        self.path=os.getcwd()
        self.pd_file = pd.read_csv(self.path+file_tweets)
        self.tweets = self.pd_file['OriginalTweet']
        self.labels = self.pd_file['Sentiment']

        assert len(self.tweets) == len(
            self.labels
        ), "Number of tweets does not match the number of labels"

    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.tweets[idx],
            self.labels[idx],
        )

        
        
tweets = CoronaTweets("test")
print(np.unique(tweets.labels))



    