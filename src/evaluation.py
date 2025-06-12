# 평가/혼동행렬/클래스지표 함수
# src/evaluation.py
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd

def evaluate_classification(y_true, y_pred):
    """
    분류 문제의 주요 성능 지표 dict 반환
    """
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'Recall': recall_score(y_true, y_pred, average='weighted'),
        'F1 Score': f1_score(y_true, y_pred, average='weighted')
    }

def confusion_df(y_true, y_pred, labels=[0,1,2]):
    """
    혼동행렬을 DataFrame 형태로 반환
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(cm, index=[f"True_{i}" for i in labels], columns=[f"Pred_{i}" for i in labels])

def class_metrics_df(y_true, y_pred, labels=[0,1,2]):
    """
    클래스별 Precision/Recall/F1 Score DataFrame 반환
    """
    return pd.DataFrame({
        'Class': labels,
        'Precision': precision_score(y_true, y_pred, average=None, labels=labels, zero_division=0),
        'Recall': recall_score(y_true, y_pred, average=None, labels=labels),
        'F1 Score': f1_score(y_true, y_pred, average=None, labels=labels)
    })
