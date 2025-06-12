# seed 고정 등 공통 함수
# src/utils.py
import numpy as np
import random
import tensorflow as tf

def set_seed(seed=42):
    """
    모든 라이브러리의 랜덤 seed 고정
    """
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    print(f"[SEED] Seed fixed to {seed}")
