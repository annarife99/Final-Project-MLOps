Natural Language Processing for Corona tweets classification:Final Project for Machine Learning Operations
==============================
This repository contains the project work of the Machine Learning Operations Course from DTU carried out by group 43. We are Alex Abades(s212784), Lluis Colomer(s213237), Anna Rifé (s212487). 

1. **Goal of the project:**
The goal of the project is to solve a classification task to determine whether a tweet being published during Corona is positive, negative or neutral by using different natural language processing (NLP). 

2. **Framework used:**
As working with NLP, we plan to use a [transformers based-model](https://github.com/huggingface/transformers) to classify the data.

3. **How are we including the framework into your project:**
Considering the transformer framework, we plan to utilize a wide range of pre-trained models that have achieved state-of-the-art performance on a variety of NLP tasks. As we are working on a tweet classification project, we may be able to achieve very high accuracy by fine-tuning one of these pre-trained models on our dataset.

4. **Data:**
We are using the Kaggle dataset of [Corona tweets for text classication] (https://www.kaggle.com/datasets/datatattle/covid-19-nlp-text-classification). The tweets have been pulled from Twitter and manual tagging has been done.

5. **Deep Learning models expected to use:**
We expect to use the transformers framework which includes highly optimized implementations of many popular NLP models, including BERT, GPT, and XLNet. These models are trained on large datasets and can process text very efficiently, making them well-suited for tasks such as our tweet classification.


Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
