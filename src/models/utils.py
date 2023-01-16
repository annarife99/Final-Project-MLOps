import numpy as np
import torch
import wandb
from model import NLPModel
from sklearn.metrics import accuracy_score, f1_score
from torch.optim import AdamW
from tqdm.notebook import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


wandb.init(project="bert-eng-model")
wandb.config = {"learning_rate": lr, "epochs": num_epochs, "batch_size": batch_size}

optimizer = AdamW(bert_model.parameters(), lr=lr)


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

    for d in data_loader:
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
        # scheduler.step()
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
