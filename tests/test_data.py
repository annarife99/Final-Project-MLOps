import pandas as pd
import os
from tests import _PATH_DATA
import torch
def test_data():
   dataset = torch.load(os.path.join(_PATH_DATA ,'processed/dataset.pth'))

   # Assert that the train split has 41157 samples and 5 features
   assert dataset['train'].shape == (41157, 5), f'Train dataset has shape {dataset["train"].shape}, expected (41157, 5)'

   # Assert that the test split has 3798 samples and 5 features
   assert dataset['test'].shape == (3798, 5), f'Test dataset has shape {dataset["test"].shape}, expected (3798, 5)'


   for datapoint in dataset['train']:
      assert len(datapoint['input_ids']) == 128
      assert len(datapoint['token_type_ids']) == 128
      assert len(datapoint['attention_mask']) == 128

   # Check if at least 5000 of each label are included in the train data
   label_count = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
   # Iterate through all datapoints in the dataset
   for datapoint in dataset['train']:
      label = datapoint['labels']
      label_count[label] += 1
   for label in label_count.keys():
      assert label_count[label] >= 5000

   # Check if at least 500 of each label are included in the test data
   label_count = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
   for datapoint in dataset['test']:
      label = datapoint['labels']
      label_count[label] += 1
   for label in label_count.keys():
      assert label_count[label] >= 500
