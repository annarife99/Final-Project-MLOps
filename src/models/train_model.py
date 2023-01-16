
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
from transformers import AutoConfig
from torch.optim import AdamW
from transformers import get_scheduler
from model import NLPModel
import wandb

_CURRENT_ROOT = os.getcwd()  # root of current file
_SRC_ROOT = os.path.dirname(_CURRENT_ROOT)  # root of src
_PROJECT_ROOT = os.path.dirname(_SRC_ROOT)  # project root
_PATH_RAW_DATA = os.path.join(_PROJECT_ROOT, "data/raw/")  # root of raw data folder
_PATH_PROCESSED_DATA = os.path.join(_PROJECT_ROOT, "data/processed/")  # root of raw data folder
import pdb; pdb.set_trace()

# from src.data.dataset import CoronaTweets, create_dataloader
#from src.data.clean_functions import preprocessText

#@hydra.main(config_path="../../config", config_name="default_config.yaml")

@hydra.main(config_path="config", config_name='config.yaml')

def main(config):#config: DictConfig):
    logger = logging.getLogger(__name__)
    logger.info("Start Training...")

    # _CURRENT_ROOT = os.getcwd()  # root of current file
    # _SRC_ROOT = os.path.dirname(_CURRENT_ROOT)  # root of src
    # _PROJECT_ROOT = os.path.dirname(_SRC_ROOT)  # project root
    # _PATH_RAW_DATA = os.path.join(_PROJECT_ROOT, "data/raw/")  # root of raw data folder
    # _PATH_PROCESSED_DATA = os.path.join(_PROJECT_ROOT, "data/processed/")  # root of raw data folder
    # import pdb; pdb.set_trace()
    hparams = config.experiment
    torch.manual_seed(hparams["seed"])

    #client = secretmanager.SecretManagerServiceClient()
    #PROJECT_ID = "dtu-mlops-project"

    #secret_id = "WANDB"
    #resource_name = f"projects/{PROJECT_ID}/secrets/{secret_id}/versions/latest"
    #response = client.access_secret_version(name=resource_name)
    #api_key = response.payload.data.decode("UTF-8")
    #os.environ["WANDB_API_KEY"] = api_key
    wandb.init(project="NLP-BERT", entity="dtu-mlops", config=config)

    gpus = 0
    if torch.cuda.is_available():
        # selects all available gpus
        print(f"Using {torch.cuda.device_count()} GPU(s) for training")
        gpus = -1
    else:
        print("Using CPU for training")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = hparams['batch_size']
    models = ["distilbert-base-uncased", "bert-base-uncased", "bert-base-cased"]
    modelName = models[2] 
    max_len = hparams['max_len']
    tokenizer = AutoTokenizer.from_pretrained(modelName)

    df_train = pd.read_csv(os.path.join(_PATH_PROCESSED_DATA, 'df_train.csv'))
    df_test = pd.read_csv(os.path.join(_PATH_PROCESSED_DATA, 'df_test.csv'))

    train_data_loader = create_dataloader(df_train, tokenizer, max_len, batch_size)
    test_data_loader = create_dataloader(df_test, tokenizer, max_len, batch_size)

    it = iter(train_data_loader)
    data_batch = next(it)
    data_batch.keys()

    print(data_batch["input_ids"].shape, data_batch["attention_mask"].shape, data_batch["targets"].shape)
    data_batch["input_ids"][0]

    id2label = {
        0: 'Extremely Negative',
        1: 'Negative',
        2: 'Neutral',
        3: 'Positive',
        4: 'Extremely Positive'
    }

    label2id = {v: k for (k, v) in id2label.items()}

    bert_config = AutoConfig.from_pretrained(modelName,
                                             num_labels=5,
                                             id2label=id2label, label2id=label2id)

    bert_model = (NLPModel
                  .from_pretrained(modelName, config=bert_config)
                  .to(device))

    lr = hparams['lr']
    optimizer = AdamW(bert_model.parameters(), lr=lr)

    num_epochs = hparams['n_epochs']

    dataset_sentAnalysis = DatasetDict()
    dataset_sentAnalysis["train"] = train_data_loader
    dataset_sentAnalysis["test"] = train_data_loader

    logging_steps = len(dataset_sentAnalysis_encoded["train"]) // batch_size
    num_training_steps = num_epochs * logging_steps
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )
    loss_fct = nn.CrossEntropyLoss().to(device)


    wandb.init(project="bert-eng-model")
    wandb.config = {
        "learning_rate": lr, "epochs": num_epochs, "batch_size": batch_size
    }



   
    model = NLPModel()#config)

    trainer = Trainer(
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
    )

    model.save_jit()


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    print(os.getcwd())
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
