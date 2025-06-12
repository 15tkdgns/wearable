# DNN 모델 생성 함수
# src/models/dnn.py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

def build_dnn(input_dim, n_classes=3, layers=[1024, 512, 256, 128], dropout=0.2):
    """
    DNN(MLP) 모델 생성 함수
    - input_dim: 입력 차원
    - n_classes: 출력 클래스 수 (softmax)
    - layers: 은닉층 유닛 개수 리스트
    - dropout: 드롭아웃 비율
    """
    model = Sequential()
    model.add(Dense(layers[0], activation='relu', input_dim=input_dim, kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Dropout(dropout))
    for units in layers[1:]:
        model.add(Dense(units, activation='relu', kernel_initializer='he_normal'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout))
    model.add(Dense(n_classes, activation='softmax'))
    return model
