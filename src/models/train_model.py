
import logging
import os
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from dotenv import find_dotenv, load_dotenv
from omegaconf import DictConfig
from pytorch_lightning import Trainer
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from datasets import Dataset , Sequence , Value , Features , ClassLabel , DatasetDict

from model import NLPModel

from src.data.dataset import CoronaTweets, create_dataloader
from src.data.clean_functions import preprocessText

#@hydra.main(config_path="../../config", config_name="default_config.yaml")
def main():#config: DictConfig):
    logger = logging.getLogger(__name__)
    logger.info("Start Training...")
    #client = secretmanager.SecretManagerServiceClient()
    #PROJECT_ID = "dtu-mlops-project"

    #secret_id = "WANDB"
    #resource_name = f"projects/{PROJECT_ID}/secrets/{secret_id}/versions/latest"
    #response = client.access_secret_version(name=resource_name)
    #api_key = response.payload.data.decode("UTF-8")
    #os.environ["WANDB_API_KEY"] = api_key
    #wandb.init(project="NLP-BERT", entity="dtu-mlops", config=config)

    """gpus = 0
    if torch.cuda.is_available():
        # selects all available gpus
        print(f"Using {torch.cuda.device_count()} GPU(s) for training")
        gpus = -1
    else:
        print("Using CPU for training")"""

    batch_size = 8
    models = ["distilbert-base-uncased", "bert-base-uncased", "bert-base-cased"]
    modelName = models[2] 
    max_len = 128
    tokenizer = AutoTokenizer.from_pretrained(modelName)
    df_train= pd.read_csv('data/processed/df_train.csv')
    df_test= pd.read_csv('data/processed/df_test.csv')

    train_data_loader = create_dataloader(df_train, tokenizer, max_len, batch_size)
    train_data_loader = create_dataloader(df_test, tokenizer, max_len, batch_size)

    dataset_sentAnalysis = DatasetDict()
    dataset_sentAnalysis["train"] = train_data_loader
    dataset_sentAnalysis["test"] = train_data_loader
    

    
   
    #model = NLPModel()#config)

    """trainer = Trainer(
        max_epochs=config.train.epochs,
        gpus=gpus,
        logger=pl.loggers.WandbLogger(project="mlops-mnist", config=config),
        val_check_interval=1.0,
        check_val_every_n_epoch=1,
        gradient_clip_val=1.0,
    )
    trainer.fit(
        model,
        train_dataloader=data_module.train_dataloader(),
        val_dataloaders=data_module.test_dataloader(),
    )"""

    #model.save_jit()


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
