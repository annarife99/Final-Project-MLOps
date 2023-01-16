# -*- coding: utf-8 -*-
import sys
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import os
import pandas as pd
import torch 
import datasets
from datasets import Dataset , Sequence , Value , Features , ClassLabel , DatasetDict
from clean_functions import preprocessBatch
from transformers import AutoTokenizer


@click.command()
#@click.argument('input_filepath', type=click.Path(exists=True))
#@click.argument('output_filepath', type=click.Path())

def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('Making final data set from raw data')

    _CURRENT_ROOT = os.path.dirname(__file__)  # root of current file
    _SRC_ROOT = os.path.dirname(_CURRENT_ROOT)  # root of src
    _PROJECT_ROOT = os.path.dirname(_SRC_ROOT)  # project root
    _PATH_RAW_DATA = os.path.join(_PROJECT_ROOT, "data/raw/")  # root of raw data folder
    _PATH_PROCESSED_DATA = os.path.join(_PROJECT_ROOT, "data/processed/")  # root of raw data folder

    pd_file_train = pd.read_csv(os.path.join(_PATH_RAW_DATA,'Corona_NLP_train.csv'),encoding='latin-1')
    pd_file_test = pd.read_csv(os.path.join(_PATH_RAW_DATA,'Corona_NLP_test.csv'),encoding='latin-1')

    pd_file_train = pd_file_train[["OriginalTweet", "Sentiment"]]
    pd_file_train.drop_duplicates(subset='OriginalTweet',inplace=True)
    pd_file_train = pd_file_train.rename({'OriginalTweet': 'Reviews'}, axis='columns')
    pd_file_train['Sentiment']=pd_file_train['Sentiment'].replace({'Neutral':2, 'Positive':3,'Extremely Positive':4, 'Extremely Negative':0,'Negative':1})
    pd_file_train['Sentiment']=pd_file_train['Sentiment'].astype(int)
    pd_file_train = pd_file_train.reset_index(drop=True)
    pd_file_train.isnull().sum()
    pd_file_train.to_csv(os.path.join(_PATH_PROCESSED_DATA, 'df_train.csv'))

    pd_file_test = pd_file_test.drop(labels=['UserName', 'ScreenName', 'Location', 'TweetAt'], axis=1)
    pd_file_test.drop_duplicates(subset='OriginalTweet',inplace=True)
    pd_file_test = pd_file_test.rename({'OriginalTweet': 'Reviews'}, axis="columns")
    pd_file_test['Sentiment']=pd_file_test['Sentiment'].replace({'Neutral':2, 'Positive':3,'Extremely Positive':4, 'Extremely Negative':0,'Negative':1})
    pd_file_test['Sentiment']=pd_file_test['Sentiment'].astype(int)
    pd_file_test = pd_file_test.reset_index(drop=True)

    pd_file_test.to_csv(os.path.join(_PATH_PROCESSED_DATA, 'df_test.csv'))
    print(pd_file_train.shape , pd_file_test.shape)

    def createDataset(df,textCol, labelCol):
        dataset_dict = {
            'text' : df[textCol],
            'labels' : df[labelCol],
        }
        sent_tags = ClassLabel(num_classes=5 , names=['Extremely Negative', 'Negative','Neutral','Positive', 'Extremely Positive'])
        return Dataset.from_dict(
            mapping = dataset_dict,
            features = Features({'text' : Value(dtype='string') , 'labels' :sent_tags})
        )

    dataset_train = createDataset(pd_file_train,"Reviews","Sentiment")
    dataset_test = createDataset(pd_file_test,"Reviews","Sentiment")
    

    dataset_sentAnalysis = DatasetDict()
    dataset_sentAnalysis["train"] = dataset_train
    dataset_sentAnalysis["test"] = dataset_test
    

    #PREPROCESS
    dataset_sentAnalysis_preprocessed = dataset_sentAnalysis.map(preprocessBatch, batched=True, batch_size=32)

    
    #TOKENIZER
    models = ["distilbert-base-uncased", "bert-base-uncased", "bert-base-cased"]
    modelName = models[2] 
    tokenizer = AutoTokenizer.from_pretrained(modelName)

    max_len = 128
    def tokenize(batch):
        return tokenizer(batch["text"], pad_to_max_length=True, truncation=True, max_length=max_len)
    
    dataset_sentAnalysis_encoded = dataset_sentAnalysis_preprocessed.map(tokenize, batched=True, batch_size=32)

    torch.save(dataset_sentAnalysis_encoded,os.path.join(_PATH_PROCESSED_DATA, 'dataset.pth'))
    

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
