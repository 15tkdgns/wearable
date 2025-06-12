# 데이터 로딩/결측치 요약 함수
# src/data_loading.py
import pandas as pd

def load_data(path):
    """
    지정한 CSV 파일 경로에서 데이터를 불러오고
    shape 정보를 출력한다.
    """
    df = pd.read_csv(path)
    print(f"[DATA] Loaded: {path}, Shape: {df.shape}")
    return df

def summarize_missing(df):
    """
    각 컬럼별 결측치 개수/비율을 반환한다.
    """
    missing_info = df.isnull().sum().reset_index()
    missing_info.columns = ['Variable', 'MissingCount']
    missing_info['MissingRatio'] = missing_info['MissingCount'] / len(df)
    print("[DATA] Missing Value Summary:\n", missing_info.head())
    return missing_info
