import torch
import torch.nn.functional as F
from torch import nn
from omegaconf import DictConfig
from pytorch_lightning import LightningModule


class NLPModel(LightningModule):
    def __init__(self,config:DictConfig):
        super().__init__()
        self.confi= config
        #self.model=

    
    def forward(self, x):
        '''  '''
        pass
        
        #return 
