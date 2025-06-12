# 결측치/이상치/인코딩 함수
# src/preprocessing.py
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

def filter_target_and_age(df, target_col='mh_PHQ_S', min_age=15):
    """
    타겟 결측치가 있는 행 제거 + min_age 미만 제거
    """
    df = df.dropna(subset=[target_col]).copy()
    df = df[df['age'] >= min_age].copy()
    return df

def fill_numeric_missing(df, numeric_cols):
    """
    숫자형 컬럼의 결측값을 평균으로 대치
    """
    imputer = SimpleImputer(strategy='mean')
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    return df

def remove_outliers_iqr(df, numeric_cols):
    """
    IQR 방법으로 이상치 제거(모든 숫자형 컬럼에 대해)
    """
    for col in numeric_cols:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        df = df[(df[col] >= lower) & (df[col] <= upper)]
    return df

def encode_categorical(df, categorical_cols):
    """
    모든 범주형 컬럼을 Label Encoding으로 변환
    """
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    return df
