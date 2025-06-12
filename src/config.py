# src/config.py
"""
실험 환경/하이퍼파라미터/경로 설정 (Git 관리 용이)
"""
import os

SEED = 42   # 전체 실험 seed (랜덤 일관성용)
# 데이터 파일 경로 (프로젝트 구조에 따라 상대경로!)
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', '241103.csv')
RESULT_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
N_POLY_DEGREE = 2         # 다항특징 차수
K_BEST_FEATURES = 11      # SelectKBest에서 뽑을 피처 개수
TEST_SIZE = 0.2           # 테스트셋 비율
SCALE_TYPE = 'StandardScaler'    # 추천: Standard, MinMax, Robust 등
OPTIMIZER = 'adam'        # 옵티마이저 (문자열/클래스 모두 가능)
BATCH_SIZE = 32
EPOCHS = 50
PATIENCE = 5
DNN_LAYERS = [1024, 512, 256, 128]
CNN_DROPOUT = 0.2
