import glob
import os

import yaml
from fastapi import FastAPI
from fastapi.responses import FileResponse

app = FastAPI()

@app.post("/conf_files/")
def conf_files(lr: float, batch_size:int, n_epochs:int, max_len:int, seed:int,name:str):

    data = {
    'lr': lr,
    'batch_size': batch_size,
    'n_epochs': n_epochs,
    'max_len': max_len,
    'seed': seed
    }

    name='../src/models/config/experiment/'+str(name)+'.yaml'

    with open(name, 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)

    return "conf file created"

