# ML 모델 생성 함수 (RF/XGB/LGB 등)
# src/models/classical.py
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
import xgboost as xgb
import lightgbm as lgb

def get_classical_models(seed=42):
    """
    여러 classical ML 모델 생성/반환
    """
    return {
        'LogisticRegression': LogisticRegression(random_state=seed, max_iter=1000),
        'RandomForest': RandomForestClassifier(random_state=seed),
        'XGBoost': xgb.XGBClassifier(random_state=seed, use_label_encoder=False, eval_metric='mlogloss'),
        'SVM': SVC(probability=True, random_state=seed),
        'AdaBoost': AdaBoostClassifier(random_state=seed),
        'LightGBM': lgb.LGBMClassifier(random_state=seed)
    }
