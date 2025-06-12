# src/models/tabnet.py
import numpy as np
from pytorch_tabnet.tab_model import TabNetClassifier

def train_tabnet(X_train, y_train, X_valid, y_valid, seed=42, max_epochs=200):
    clf = TabNetClassifier(seed=seed, verbose=0)
    clf.fit(
        X_train.values, y_train.values,
        eval_set=[(X_valid.values, y_valid.values)],
        eval_metric=['accuracy'],
        max_epochs=max_epochs,
        patience=20,
        batch_size=1024,
        virtual_batch_size=128
    )
    return clf