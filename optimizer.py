# coding:utf-8
import os, sys
import os.path as osp
import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from functools import reduce
from torch.optim import AdamW

def define_scheduler(optimizer, params):
    print(params)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=params.get("max_lr", 2e-4),
        epochs=params.get("epochs", 200),
        steps_per_epoch=params.get("steps_per_epoch", 1000),
        pct_start=params.get("pct_start", 0.0),
        div_factor=1,
        final_div_factor=1,
    )

    return scheduler


def build_optimizer(model, scheduler_params_dict):
    optim = AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-4,
        betas=(0.0, 0.99),
        eps=1e-9,
    )
    scheduler = define_scheduler(optim, scheduler_params_dict)
    return optim, scheduler
