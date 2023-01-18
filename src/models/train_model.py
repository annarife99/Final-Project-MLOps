import logging
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict

import hydra
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import wandb
from datasets import ClassLabel, Dataset, DatasetDict, Features, Sequence, Value
from dotenv import find_dotenv, load_dotenv
from model import NLPModel
from omegaconf import DictConfig
from pytorch_lightning import Trainer

# from src.data.dataset import CoronaTweets, create_dataloader
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.optim import AdamW
from tqdm import tqdm
from tqdm.notebook import tqdm
from transformers import AutoConfig, AutoTokenizer, get_scheduler
#from src.data.dataset import CoronaTweets, create_dataloader
# from src.data.clean_functions import preprocessText

# @hydra.main(config_path="../../config", config_name="default_config.yaml")


@hydra.main(config_path="config", config_name="config.yaml")
@hydra.main(config_path="config", config_name="config.yaml")
def main(config: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Start Training...")

    _CURRENT_ROOT = os.path.abspath(os.path.dirname(__file__))  # root of current file
    _SRC_ROOT = os.path.dirname(_CURRENT_ROOT)  # root of src
    _PROJECT_ROOT = os.path.dirname(_SRC_ROOT)  # project root
    _PATH_RAW_DATA = os.path.join(_PROJECT_ROOT, "data/raw/")  # root of raw data folder
    _PATH_PROCESSED_DATA = os.path.join(_PROJECT_ROOT, "data/processed/")  # root of raw data folder
    sys.path.append(_PROJECT_ROOT)
    from src.data.dataset import CoronaTweets, create_dataloader

    hparams = config.experiment
    torch.manual_seed(hparams["seed"])

    # client = secretmanager.SecretManagerServiceClient()
    # PROJECT_ID = "dtu-mlops-project"

    # secret_id = "WANDB"
    # resource_name = f"projects/{PROJECT_ID}/secrets/{secret_id}/versions/latest"
    # response = client.access_secret_version(name=resource_name)
    # api_key = response.payload.data.decode("UTF-8")
    # os.environ["WANDB_API_KEY"] = api_key

    # api_key='366e12344bd2ff60ee203fae40c62940b249e3ff'
    wandb.init(project="NLP-BERT", entity="ml-operations")
    if torch.cuda.is_available():
        # selects all available gpus
        print(f"Using {torch.cuda.device_count()} GPU(s) for training")
        gpus = -1
    else:
        print("Using CPU for training")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = hparams["batch_size"]
    models = ["distilbert-base-uncased", "bert-base-uncased", "bert-base-cased"]
    modelName = models[2]
    max_len = hparams["max_len"]
    tokenizer = AutoTokenizer.from_pretrained(modelName)

    df_train = pd.read_csv(os.path.join(_PATH_PROCESSED_DATA, "df_train.csv"), nrows=100)
    df_test = pd.read_csv(os.path.join(_PATH_PROCESSED_DATA, "df_test.csv"), nrows=20)

    train_data_loader = create_dataloader(df_train, tokenizer, max_len, batch_size)
    test_data_loader = create_dataloader(df_test, tokenizer, max_len, batch_size)

    it = iter(train_data_loader)
    data_batch = next(it)
    data_batch.keys()

    print(
        data_batch["input_ids"].shape,
        data_batch["attention_mask"].shape,
        data_batch["targets"].shape,
    )
    data_batch["input_ids"][0]

    id2label = {
        0: "Extremely Negative",
        1: "Negative",
        2: "Neutral",
        3: "Positive",
        4: "Extremely Positive",
    }

    label2id = {v: k for (k, v) in id2label.items()}

    bert_config = AutoConfig.from_pretrained(
        modelName, num_labels=5, id2label=id2label, label2id=label2id
    )

    bert_model = NLPModel.from_pretrained(modelName, config=bert_config).to(device)

    lr = hparams["lr"]
    optimizer = AdamW(bert_model.parameters(), lr=lr)

    num_epochs = hparams["n_epochs"]
    print("EPOCHS", num_epochs)

    dataset_sentAnalysis = DatasetDict()
    dataset_sentAnalysis["train"] = train_data_loader
    dataset_sentAnalysis["test"] = train_data_loader

    dataset_sentAnalysis_encoded = torch.load(os.path.join(_PATH_PROCESSED_DATA, "dataset.pth"))
    logging_steps = len(dataset_sentAnalysis_encoded["train"]) // batch_size
    num_training_steps = num_epochs * logging_steps
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    loss_fct = nn.CrossEntropyLoss().to(device)

    wandb.config = {"learning_rate": lr, "epochs": num_epochs, "batch_size": batch_size}

    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        f1 = f1_score(labels, preds, average="weighted")
        acc = accuracy_score(labels, preds)
        return {"accuracy": acc, "f1": f1}

    def eval_op(model, data_loader, loss_fn, n_examples):
        model.eval()

        losses = []
        correct_predictions = 0

        with torch.no_grad():
            for d in data_loader:
                input_ids = d["input_ids"].to(device)
                attention_mask = d["attention_mask"].to(device)
                targets = d["targets"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.max(outputs.logits, dim=1)
                loss = loss_fn(outputs.logits, targets)
                correct_predictions += torch.sum(preds.indices == targets)
                losses.append(loss.item())
        wandb.log(
            {
                "loss-eval": np.mean(losses),
                "accuracy-eval": correct_predictions.double(),
                "learning-rate": optimizer.param_groups[0]["lr"],
            }
        )
        return correct_predictions.double() / n_examples, np.mean(losses)

    def train_epoch(model, data_loader, loss_fn, optimizer, n_examples, scheduler=None):
        # put the model in training mode > dropout is considered for exp
        model.train()
        losses = []
        correct_predictions = 0
        print(len(data_loader))
        i = 0

        for d in data_loader:
            print(i)
            i += 1
            input_ids = d["input_ids"].to(device)  # bs*classes
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.max(outputs.logits, dim=1)

            # the loss has grad function
            loss = loss_fn(outputs.logits, targets)
            correct_predictions += torch.sum(preds.indices == targets)
            losses.append(loss.item())
            loss.backward()

            # avoid exploding gradient - gradient clipping
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        wandb.log(
            {
                "loss-train": np.mean(losses),
                "accuracy-train": correct_predictions.double(),
                "learning-rate": optimizer.param_groups[0]["lr"],
            }
        )

        # return accuracy and loss
        return correct_predictions.double() / n_examples, np.mean(losses)

    history = defaultdict(list)
    best_accuracy = 0
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("-" * 10)

        train_acc, train_loss = train_epoch(
            bert_model,
            train_data_loader,
            loss_fct,
            optimizer,
            len(df_train),
            scheduler=lr_scheduler,
        )
        print(f"Train loss {train_loss} accuracy {train_acc}")

        val_acc, val_loss = eval_op(bert_model, test_data_loader, loss_fct, len(df_test))
        print(f"Val loss {val_loss} accuracy {val_acc}")
        history["train_acc"].append(train_acc)
        history["train_loss"].append(train_loss)
        history["val_acc"].append(val_acc)
        history["val_loss"].append(val_loss)

        if val_acc > best_accuracy:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": bert_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": train_loss,
                },
                f"./bert-eng.bin",
            )
            best_accuracy = val_acc



if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
    print("End")
