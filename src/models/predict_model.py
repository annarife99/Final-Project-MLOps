import glob
import logging
import os
from pathlib import Path
from time import time

import hydra
import matplotlib.pyplot as plt
import numpy as np
import omegaconf
import pandas as pd
import seaborn as sns
import torch.quantization
import wandb
from dotenv import find_dotenv, load_dotenv
from model import NLPModel
from omegaconf import DictConfig
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.optim import AdamW
from transformers import AutoConfig, AutoTokenizer

from src.data.dataset import CoronaTweets, create_dataloader


@hydra.main(config_path="config", config_name="config.yaml")
def main(config: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Executing predict model script...")

    _CURRENT_ROOT = os.path.abspath(os.path.dirname(__file__))  # root of current file
    _SRC_ROOT = os.path.dirname(_CURRENT_ROOT)  # root of src
    _PROJECT_ROOT = os.path.dirname(_SRC_ROOT)  # project root
    _PATH_RAW_DATA = os.path.join(_PROJECT_ROOT, "data/raw/")  # root of raw data folder
    _PATH_PROCESSED_DATA = os.path.join(_PROJECT_ROOT, "data/processed/")  # root of raw data folder

    hparams = config.experiment
    torch.manual_seed(hparams["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models = ["distilbert-base-uncased", "bert-base-uncased", "bert-base-cased"]
    modelName = models[2]

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

    checkpoint = torch.load(
        os.path.join(_SRC_ROOT, "models/outputs/2023-01-16/15-32-58/bert-eng.bin")
    )
    bert_model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    train_loss = checkpoint["loss"]

    def eval_op(model, data_loader, loss_fn, n_examples):
        model.eval()

        losses = []
        correct_predictions = 0

        wandb.init(project="NLP-BERT", entity="ml-operations")

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

    df_test = pd.read_csv(os.path.join(_PATH_PROCESSED_DATA, "df_test.csv"), nrows=100)
    batch_size = hparams["batch_size"]
    max_len = hparams["max_len"]
    models = ["distilbert-base-uncased", "bert-base-uncased", "bert-base-cased"]
    modelName = models[2]
    tokenizer = AutoTokenizer.from_pretrained(modelName)

    test_data_loader = create_dataloader(df_test, tokenizer, max_len, batch_size)
    loss_fct = nn.CrossEntropyLoss().to(device)

    # the best model checkpoint saves in path > /content/drive/MyDrive/bert-eng-3.bin
    test_acc, test_loss = eval_op(bert_model, test_data_loader, loss_fct, len(df_test))
    print(test_acc, test_loss)

    def get_reviews(model, data_loader):

        model = model.eval()
        review_texts = []
        predictions = []
        prediction_probs = []
        real_values = []

        with torch.no_grad():
            for d in data_loader:
                texts = d["review_text"]
                input_ids = d["input_ids"].to(device)
                attention_mask = d["attention_mask"].to(device)
                targets = d["targets"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                _, preds = torch.max(outputs.logits, dim=1)

                review_texts.extend(texts)
                predictions.extend(preds)
                prediction_probs.extend(outputs.logits)
                real_values.extend(targets)

        # convert the list of tensors to a single tensor
        # review_texts is a list of strings not tensors
        predictions = torch.stack(predictions).cpu()
        prediction_probs = torch.stack(prediction_probs).cpu()
        real_values = torch.stack(real_values).cpu()

        return review_texts, predictions, prediction_probs, real_values

    y_review_texts, y_pred, y_probs, y_test = get_reviews(bert_model, test_data_loader)

    class_names = ["Extremely Negative", "Negative", "Neutral", "Positive", "Extremely Positive"]
    print(classification_report(y_test, y_pred, target_names=class_names))

    def show_confusion_matrix(confusion_matrix):
        hmap = sns.heatmap(confusion_matrix, annot=True, fmt=".2f", cmap="Blues")
        hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha="right")
        hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha="right")
        plt.ylabel("True sentiment")
        plt.xlabel("Predicted sentiment")
        plt.show()

    cm = confusion_matrix(y_test, y_pred, normalize="true")
    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
    show_confusion_matrix(df_cm)


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
