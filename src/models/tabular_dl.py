# src/models/tabular_dl.py
from node import NODE  # assuming NODE repo installed or vendored
from saint import SAINT
from ft_transformer import FT_Transformer
import torch
import torch.nn as nn

def build_node(input_dim, output_dim, config=None):
    # config: dict of hyperparameters
    model = NODE(input_dim=input_dim, num_outputs=output_dim, **(config or {}))
    return model


def build_saint(input_dim, output_dim, config=None):
    model = SAINT(input_dim=input_dim, output_dim=output_dim, **(config or {}))
    return model


def build_ft_transformer(input_dim, output_dim, config=None):
    model = FT_Transformer(input_dim=input_dim, num_classes=output_dim, **(config or {}))
    return model