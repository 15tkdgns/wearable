# 1D CNN 모델 생성 함수
# src/models/cnn.py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D, Dense, Dropout, BatchNormalization

def build_cnn(input_length, n_classes=3, dropout=0.2):
    """
    1D CNN 모델 생성 함수
    - input_length: 입력 시퀀스 길이 (= 피처 수)
    - n_classes: 출력 클래스 수 (softmax)
    """
    model = Sequential()
    model.add(Conv1D(128, 3, activation='relu', input_shape=(input_length,1), padding='same'))
    model.add(BatchNormalization())
    model.add(Dropout(dropout))
    model.add(Conv1D(64, 3, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Dropout(dropout))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(n_classes, activation='softmax'))
    return model
