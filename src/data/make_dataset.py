# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import os
import pandas as pd
import torch 


@click.command()
#@click.argument('input_filepath', type=click.Path(exists=True))
#@click.argument('output_filepath', type=click.Path())

def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('Making final data set from raw data')

    path=os.getcwd()
    pd_file_train = pd.read_csv(path+'/data/raw/Corona_NLP_train.csv',encoding='ISO-8859-1')
    pd_file_test = pd.read_csv(path+'/data/raw/Corona_NLP_test.csv')

    #CHANGE LABELS
    pd_file_train.loc[pd_file_train['Sentiment'] == "Extremely Negative", 'Sentiment'] = 0
    pd_file_train.loc[pd_file_train['Sentiment'] == "Negative", 'Sentiment'] = 1
    pd_file_train.loc[pd_file_train['Sentiment'] == "Neutral", 'Sentiment'] = 2
    pd_file_train.loc[pd_file_train['Sentiment'] == "Positive", 'Sentiment'] = 3
    pd_file_train.loc[pd_file_train['Sentiment'] == "Extremely Positive", 'Sentiment'] = 4

    pd_file_test.loc[pd_file_test['Sentiment'] == "Extremely Negative", 'Sentiment'] = 0
    pd_file_test.loc[pd_file_test['Sentiment'] == "Negative", 'Sentiment'] = 1
    pd_file_test.loc[pd_file_test['Sentiment'] == "Neutral", 'Sentiment'] = 2
    pd_file_test.loc[pd_file_test['Sentiment'] == "Positive", 'Sentiment'] = 3
    pd_file_test.loc[pd_file_test['Sentiment'] == "Extremely Positive", 'Sentiment'] = 4


    train_dic={}
    train_dic['labels']= pd_file_train['Sentiment'].to_numpy()
    train_dic['tweets']= pd_file_train['OriginalTweet'].to_numpy()

    test_dic={}
    test_dic['labels']= pd_file_test['Sentiment'].to_numpy()
    test_dic['tweets']= pd_file_test['OriginalTweet'].to_numpy()
    
    torch.save(train_dic, path+'/data/processed/train.pth')
    torch.save(test_dic, path+'/data/processed/test.pth')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
