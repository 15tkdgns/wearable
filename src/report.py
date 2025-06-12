# 엑셀 저장 함수
# src/report.py
import pandas as pd
import os

def save_to_excel(report_dict, output_filename):
    """
    여러 DataFrame(dict) → 엑셀 파일의 여러 시트로 저장
    """
    with pd.ExcelWriter(output_filename) as writer:
        for sheet, df in report_dict.items():
            df.to_excel(writer, sheet_name=sheet, index=True)
    print(f"[REPORT] 결과가 '{output_filename}' 파일로 저장됨")
