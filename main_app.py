import pandas as pd
import numpy as np
import torch
import sys
from transformers import AutoTokenizer
from transformers import get_scheduler
import os
from pathlib import Path
from torch import nn
from transformers import AutoConfig
from torch.optim import AdamW
from datasets import DatasetDict
from collections import defaultdict
import wandb
import logging
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

from fastapi import FastAPI
from fastapi import UploadFile, File

from http import HTTPStatus

app = FastAPI()

@app.post("/train_model/")

async def train_model(lr: float, batch_size:int, n_epochs:int, max_len:int, seed:int, ntrain: int, ntest:int):
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
    from src.models.model import NLPModel
    torch.manual_seed(seed)

    wandb.init(project="NLP-BERT", entity="ml-operations")
    if torch.cuda.is_available():
        # selects all available gpus
        print(f"Using {torch.cuda.device_count()} GPU(s) for training")
        gpus = -1
    else:
        print("Using CPU for training")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models = ["distilbert-base-uncased", "bert-base-uncased", "bert-base-cased"]
    modelName = models[2] 
    tokenizer = AutoTokenizer.from_pretrained(modelName)
    print(_CURRENT_ROOT)

    df_train = pd.read_csv(os.path.join(_CURRENT_ROOT, 'data/processed/df_train.csv'),nrows=ntrain)
    df_test = pd.read_csv(os.path.join(_CURRENT_ROOT, 'data/processed/df_test.csv'),nrows=ntest)

    train_data_loader = create_dataloader(df_train, tokenizer, max_len, batch_size)
    test_data_loader = create_dataloader(df_test, tokenizer, max_len, batch_size)

    it = iter(train_data_loader)
    data_batch = next(it)
    data_batch.keys()

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

    optimizer = AdamW(bert_model.parameters(), lr=lr)

    dataset_sentAnalysis = DatasetDict()
    dataset_sentAnalysis["train"] = train_data_loader
    dataset_sentAnalysis["test"] = train_data_loader

    dataset_sentAnalysis_encoded=torch.load(os.path.join(_CURRENT_ROOT,'data/processed/dataset.pth'))
    logging_steps = len(dataset_sentAnalysis_encoded["train"]) // batch_size
    num_training_steps = n_epochs * logging_steps
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )
    loss_fct = nn.CrossEntropyLoss().to(device)

    wandb.config = {
        "learning_rate": lr, "epochs": n_epochs, "batch_size": batch_size
    }


    def eval_op(model, data_loader, loss_fn, n_examples):
        model.eval()

        losses = []
        correct_predictions = 0

        with torch.no_grad():
            for d in data_loader:
                input_ids = d["input_ids"].to(device)
                attention_mask = d["attention_mask"].to(device)
                targets = d["targets"].to(device)
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                preds = torch.max(outputs.logits, dim=1)
                loss = loss_fn(outputs.logits, targets)
                correct_predictions += torch.sum(preds.indices == targets)
                losses.append(loss.item())
        
        return correct_predictions.double() / n_examples, np.mean(losses)

    def train_epoch(
            model,
            data_loader,
            loss_fn,
            optimizer,
            n_examples,
            scheduler=None
    ):
        # put the model in training mode > dropout is considered for exp
        model.train()
        losses = []
        correct_predictions = 0
        i=0

        for d in data_loader:
            i+=1
            input_ids = d["input_ids"].to(device)  # bs*classes
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
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

            wandb.log({
            "loss-train": np.mean(losses),
            "accuracy-train": correct_predictions.double(),
            "learning-rate": optimizer.param_groups[0]['lr']
        })
        # return accuracy and loss
        return correct_predictions.double() / n_examples, np.mean(losses)

    history = defaultdict(list)
    best_accuracy = 0
    for epoch in range(n_epochs):
        train_acc, train_loss = train_epoch(
            bert_model,
            train_data_loader,
            loss_fct,
            optimizer,
            len(df_train),
            scheduler=lr_scheduler
        )

        val_acc, val_loss = eval_op(
            bert_model,
            test_data_loader,
            loss_fct,
            len(df_test)
        )
    
        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)

        if val_acc > best_accuracy:
            torch.save({
                'epoch': epoch,
                'model_state_dict': bert_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss
            }, f'./bert-eng.bin')
            best_accuracy = val_acc

    return "Training Finished"




@app.post("/predict_model/")

async def predict_model(lr: float, batch_size:int, max_len:int, seed:int, ntest:int):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Executing predict model script...")

    _CURRENT_ROOT = os.path.abspath(os.path.dirname(__file__))  # root of current file
    _SRC_ROOT = os.path.dirname(_CURRENT_ROOT)  # root of src
    _PROJECT_ROOT = os.path.dirname(_SRC_ROOT)  # project root
    sys.path.append(_PROJECT_ROOT)
    from src.data.dataset import CoronaTweets, create_dataloader
    from src.models.model import NLPModel

    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models = ["distilbert-base-uncased", "bert-base-uncased", "bert-base-cased"]
    modelName = models[2]

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
    
    optimizer = AdamW(bert_model.parameters(), lr=lr)

    checkpoint = torch.load('bert-eng.bin')
    bert_model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    train_loss = checkpoint['loss']

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
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                preds = torch.max(outputs.logits, dim=1)
                loss = loss_fn(outputs.logits, targets)
                correct_predictions += torch.sum(preds.indices == targets)
                losses.append(loss.item())
        wandb.log({
            "loss-eval": np.mean(losses),
            "accuracy-eval": correct_predictions.double(),
            "learning-rate": optimizer.param_groups[0]['lr']
        })
        return correct_predictions.double() / n_examples, np.mean(losses)

    df_test = pd.read_csv(os.path.join(_CURRENT_ROOT, 'data/processed/df_test.csv'),nrows=ntest)
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

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
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

    class_names = ['Extremely Negative', 'Negative', 'Neutral', 'Positive', 'Extremely Positive']
    print(classification_report(y_test, y_pred, target_names=class_names))

    def show_confusion_matrix(confusion_matrix):
        hmap = sns.heatmap(confusion_matrix, annot=True, fmt=".2f", cmap="Blues")
        hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
        hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha='right')
        plt.ylabel('True sentiment')
        plt.xlabel('Predicted sentiment')
        plt.show()

    cm = confusion_matrix(y_test, y_pred, normalize="true")
    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
    show_confusion_matrix(df_cm)

    return 'Prediction Completed'
