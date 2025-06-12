# 다항특징, SelectKBest 등
# src/feature_engineering.py
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_classif
import pandas as pd

def add_polynomial_features(df, numeric_cols, degree=2):
    """
    다항 특징 생성 (기본: 2차)
    - numeric_cols에 대해서만 다항식 생성, 기존 df와 병합
    """
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    poly_features = poly.fit_transform(df[numeric_cols])
    poly_names = poly.get_feature_names_out(numeric_cols)
    poly_df = pd.DataFrame(poly_features, columns=poly_names, index=df.index)
    df_poly = pd.concat([df, poly_df], axis=1)
    return df_poly, poly_names

def select_kbest_features(X, y, k=11):
    """
    ANOVA F-test 기반 SelectKBest로 k개 피처 선택
    (mask: 선택된 컬럼 불리언 배열)
    """
    selector = SelectKBest(score_func=f_classif, k=k)
    X_selected = selector.fit_transform(X, y)
    mask = selector.get_support()
    return X_selected, mask, selector
