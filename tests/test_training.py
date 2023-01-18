import os
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
from datasets import ClassLabel, Dataset, DatasetDict, Features, Sequence, Value
from src.models.model import NLPModel
from torch import nn
from torch.optim import AdamW
from transformers import AutoConfig, AutoTokenizer, get_scheduler
from src.data.dataset import CoronaTweets, create_dataloader
from tests import _PATH_DATA

def test_main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 8
    models = ["distilbert-base-uncased", "bert-base-uncased", "bert-base-cased"]
    modelName = models[2]
    max_len = 128
    tokenizer = AutoTokenizer.from_pretrained(modelName)

    df_train = pd.read_csv(os.path.join(_PATH_DATA, "processed/df_train.csv"), nrows=16)
    df_test = pd.read_csv(os.path.join(_PATH_DATA, "processed/df_test.csv"), nrows=8)

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

    lr = 2e-5
    optimizer = AdamW(bert_model.parameters(), lr=lr)

    num_epochs = 1

    dataset_sentAnalysis = DatasetDict()
    dataset_sentAnalysis["train"] = train_data_loader
    dataset_sentAnalysis["test"] = train_data_loader

    dataset_sentAnalysis_encoded = torch.load(os.path.join(_PATH_DATA, "processed/dataset.pth"))
    logging_steps = len(dataset_sentAnalysis_encoded["train"]) // batch_size
    num_training_steps = num_epochs * logging_steps
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    loss_fct = nn.CrossEntropyLoss().to(device)
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

        # return accuracy and loss
        return correct_predictions.double() / n_examples, np.mean(losses)

    history = defaultdict(list)
    best_accuracy = 0
    for epoch in range(num_epochs):
        train_acc, train_loss = train_epoch(
            bert_model,
            train_data_loader,
            loss_fct,
            optimizer,
            len(df_train),
            scheduler=lr_scheduler,
        )
        val_acc, val_loss = eval_op(bert_model, test_data_loader, loss_fct, len(df_test))
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

    assert torch.all(best_accuracy > 0.2)
