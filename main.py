import os
import pandas as pd
from datetime import datetime

from src.config import *
from src.data_loading import load_data, summarize_missing
from src.preprocessing import filter_target_and_age, fill_numeric_missing, remove_outliers_iqr, encode_categorical
from src.feature_engineering import add_polynomial_features, select_kbest_features
from src.models.dnn import build_dnn
from src.models.cnn import build_cnn
from src.models.classical import get_classical_models
from src.models.tabnet import train_tabnet   # << TabNet 추가
from src.evaluation import evaluate_classification, confusion_df, class_metrics_df
from src.report import save_to_excel
from src.utils import set_seed
from src.models.tabular_dl import build_node, build_saint, build_ft_transformer

def group_phq_score(score):
    if score <= 4: return 0
    elif score <= 9: return 1
    else: return 2

def main():
    # 1. Seed 고정
    set_seed(SEED)

    # 2~9. 데이터 준비(동일, 생략)

    df = load_data(DATA_PATH)
    summarize_missing(df)
    df = filter_target_and_age(df)
    numeric_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()
    if 'mh_PHQ_S' in numeric_cols: numeric_cols.remove('mh_PHQ_S')
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    df = fill_numeric_missing(df, numeric_cols)
    df = remove_outliers_iqr(df, numeric_cols)
    df = encode_categorical(df, categorical_cols)
    df_poly, poly_names = add_polynomial_features(df, numeric_cols, degree=N_POLY_DEGREE)
    df_poly['mh_PHQ_S_grouped'] = df_poly['mh_PHQ_S'].apply(group_phq_score)
    X = df_poly.drop(['mh_PHQ_S', 'mh_PHQ_S_grouped'], axis=1)
    y = df_poly['mh_PHQ_S_grouped']
    from sklearn.impute import SimpleImputer
    X = pd.DataFrame(SimpleImputer(strategy='mean').fit_transform(X), columns=X.columns)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED)
    from imblearn.over_sampling import SMOTE
    X_train_res, y_train_res = SMOTE(random_state=SEED).fit_resample(X_train, y_train)
    X_train_sel, mask, _ = select_kbest_features(X_train_res, y_train_res, k=K_BEST_FEATURES)
    X_test_sel = X_test.iloc[:, mask]
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_sel = scaler.fit_transform(X_train_sel)
    X_test_sel = scaler.transform(X_test_sel)

    # --- DNN 실험 ---
    dnn_model = build_dnn(input_dim=X_train_sel.shape[1])
    dnn_model.compile(optimizer=OPTIMIZER, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    dnn_model.fit(X_train_sel, y_train_res, validation_split=0.2, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=0)
    dnn_pred = dnn_model.predict(X_test_sel).argmax(axis=1)
    dnn_metrics = evaluate_classification(y_test, dnn_pred)

    # --- 1D CNN 실험 ---
    X_train_cnn = X_train_sel.reshape(-1, X_train_sel.shape[1], 1)
    X_test_cnn = X_test_sel.reshape(-1, X_test_sel.shape[1], 1)
    cnn_model = build_cnn(input_length=X_train_cnn.shape[1])
    cnn_model.compile(optimizer=OPTIMIZER, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    cnn_model.fit(X_train_cnn, y_train_res, validation_split=0.2, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=0)
    cnn_pred = cnn_model.predict(X_test_cnn).argmax(axis=1)
    cnn_metrics = evaluate_classification(y_test, cnn_pred)

    # --- Classical ML 실험 ---
    ml_models = get_classical_models(SEED)
    ml_results = {}
    for name, model in ml_models.items():
        model.fit(X_train_sel, y_train_res)
        y_pred = model.predict(X_test_sel)
        ml_results[name] = evaluate_classification(y_test, y_pred)

    # --- TabNet 실험 ---
    # (TabNet은 validation 데이터를 요구. 여기선 test셋으로 사용)
    tabnet_model = train_tabnet(
        pd.DataFrame(X_train_sel), y_train_res,
        pd.DataFrame(X_test_sel), y_test,
        seed=SEED, max_epochs=EPOCHS
    )
    tabnet_pred = tabnet_model.predict(X_test_sel)
    tabnet_metrics = evaluate_classification(y_test, tabnet_pred)
    
    # --- NODE 실험 ---
    node_model = build_node(input_dim=X_train_sel.shape[1], output_dim=3)
    node_model.fit(X_train_sel, y_train_res)
    node_pred = node_model.predict(X_test_sel)
    node_metrics = evaluate_classification(y_test, node_pred)

    # --- SAINT 실험 ---
    saint_model = build_saint(input_dim=X_train_sel.shape[1], output_dim=3)
    saint_model.fit(X_train_sel, y_train_res)
    saint_pred = saint_model.predict(X_test_sel)
    saint_metrics = evaluate_classification(y_test, saint_pred)

    # --- FT-Transformer 실험 ---
    ft_model = build_ft_transformer(input_dim=X_train_sel.shape[1], output_dim=3)
    ft_model.fit(X_train_sel, y_train_res)
    ft_pred = ft_model.predict(X_test_sel)
    ft_metrics = evaluate_classification(y_test, ft_pred)


    # --- 리포트 파일명/파라미터/엑셀 저장 ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    param_str = (
        f"DNN{DNN_LAYERS}_CNNdrop{CNN_DROPOUT}_Poly{N_POLY_DEGREE}_"
        f"KBest{K_BEST_FEATURES}_Scaler{SCALE_TYPE}_Opt{OPTIMIZER}_Seed{SEED}"
    )
    os.makedirs(RESULT_DIR, exist_ok=True)
    output_filename = os.path.join(RESULT_DIR, f"report_{timestamp}_{param_str}.xlsx")

    param_dict = {
        'DNN_Layers': str(DNN_LAYERS),
        'CNN_Dropout': CNN_DROPOUT,
        'PolyDegree': N_POLY_DEGREE,
        'KBestFeatures': K_BEST_FEATURES,
        'Scaler': SCALE_TYPE,
        'Optimizer': OPTIMIZER,
        'Seed': SEED,
        'BatchSize': BATCH_SIZE,
        'Epochs': EPOCHS,
        'TestSize': TEST_SIZE,
    }

    report_dict = {
        'DNN_Metrics': pd.DataFrame([dnn_metrics]),
        'CNN_Metrics': pd.DataFrame([cnn_metrics]),
        'DNN_Confusion': confusion_df(y_test, dnn_pred),
        'CNN_Confusion': confusion_df(y_test, cnn_pred),
        'DNN_ClassMetrics': class_metrics_df(y_test, dnn_pred),
        'CNN_ClassMetrics': class_metrics_df(y_test, cnn_pred),
        "Experiment_Params": pd.DataFrame(list(param_dict.items()), columns=["Parameter", "Value"]),
        'TabNet_Metrics': pd.DataFrame([tabnet_metrics]),
        'NODE_Metrics': pd.DataFrame([node_metrics]),
+       'SAINT_Metrics': pd.DataFrame([saint_metrics]),
+       'FTTransformer_Metrics': pd.DataFrame([ft_metrics])
     
    }
    for k, v in ml_results.items():
        report_dict[f"{k}_Metrics"] = pd.DataFrame([v])
    save_to_excel(report_dict, output_filename)

if __name__ == '__main__':
    main()
