import os

import matplotlib.pyplot as plt
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


THIS_FOLDER = os.path.dirname(__file__)
DATA_FOLDER = os.path.join(THIS_FOLDER, "dados")

FOLDERS = {
    "normal": os.path.join(DATA_FOLDER, "normal"),
    "desbalanceada": os.path.join(DATA_FOLDER, "helice_desbalanceada"),
    "quebrada": os.path.join(DATA_FOLDER, "helice_quebrada"),
    "invertida": os.path.join(DATA_FOLDER, "rotacao_invertida"),
}

DROP_COLUMNS = ["Celula3_Kgf", "Tempo", "RPM", "Potencia_W"]
MODEL_PATH = os.path.join(THIS_FOLDER, "KNN_bestModel.pkl")
CONFUSION_MATRIX_PATH = os.path.join(THIS_FOLDER, "KNN_confusion_matrix.png")
ARTICLE_CONFUSION_MATRIX_PATH = os.path.join(
    THIS_FOLDER,
    "artigo",
    "imgs",
    "knn_confusion_matrix.png",
)
FEATURES_PATH = os.path.join(THIS_FOLDER, "KNN_features.csv")
FEATURE_NAMES_PATH = os.path.join(THIS_FOLDER, "KNN_feature_names.txt")
METRICS_TXT_PATH = os.path.join(THIS_FOLDER, "KNN_metrics.txt")
METRICS_CSV_PATH = os.path.join(THIS_FOLDER, "KNN_metrics.csv")


def transitory_stacionary(df):
    loc_i = df.columns.get_loc("Corrente_A") + 1

    q1 = df["Corrente_A"].quantile(0.30)
    q3 = df["Corrente_A"].quantile(0.70)
    df_40 = df[(df["Corrente_A"] >= q1) & (df["Corrente_A"] <= q3)]

    flag_stac = False
    flag_transitory_1 = True
    transitory_1 = 0
    transitory_2 = None
    mean = df_40["Corrente_A"].mean()

    for idx, row in enumerate(df.itertuples()):
        corrente = abs(row[loc_i])

        if 0.99 * mean <= corrente <= 1.3 * mean and flag_transitory_1:
            transitory_1 = idx + 12
            flag_stac = True
            flag_transitory_1 = False

        if flag_stac and 0.3 * mean <= corrente <= 0.82 * mean:
            transitory_2 = idx - 8
            break

    df_0 = df.iloc[:transitory_1]

    if transitory_2 is not None:
        df_1 = df.iloc[transitory_1:transitory_2]
        df_2 = df.iloc[transitory_2:]
    else:
        df_1 = df.iloc[transitory_1:]
        df_2 = None

    return df_0, df_1, df_2


def window_statistics(df, condicao, arquivo):
    row = {"condicao": condicao, "arquivo": arquivo}

    for coluna in df.columns:
        valores = df[coluna].dropna()

        if valores.empty:
            continue

        rms = np.sqrt(np.mean(valores**2))
        val_max = valores.max()
        val_min = valores.min()
        q75, q25 = np.percentile(valores, [75, 25])

        row[f"{coluna}_media"] = valores.mean()
        row[f"{coluna}_mediana"] = valores.median()
        row[f"{coluna}_desvio_padrao"] = valores.std()
        row[f"{coluna}_skew"] = valores.skew()
        row[f"{coluna}_kurtosis"] = valores.kurtosis()
        row[f"{coluna}_iqr"] = q75 - q25
        row[f"{coluna}_rms"] = rms
        row[f"{coluna}_energia"] = np.sum(valores**2)
        row[f"{coluna}_pico_a_pico"] = val_max - val_min
        row[f"{coluna}_fator_crista"] = np.max(np.abs(valores)) / rms if rms > 0 else 0

    return row


def collect_files():
    files_with_class = []

    for condicao, folder in FOLDERS.items():
        files = [
            os.path.join(folder, name)
            for name in os.listdir(folder)
            if name.endswith(".xlsx")
        ]

        print(f"\n================== {condicao} ==================")
        print(f"Arquivos encontrados: {len(files)}")

        for file_path in files:
            files_with_class.append((file_path, condicao))

    return files_with_class


def extract_features_from_files(files_with_class, window_length=50, step=50):
    dados_extraidos = []

    for file_path, condicao in files_with_class:
        try:
            df = pd.read_excel(file_path)
            df = df.drop(columns=DROP_COLUMNS, errors="ignore")

            _, df_regime, _ = transitory_stacionary(df)
            df_regime = df_regime.dropna().reset_index(drop=True)

            if df_regime.empty:
                print(f"Aviso: regime permanente vazio em {os.path.basename(file_path)}")
                continue

            for inicio in range(0, len(df_regime) - window_length + 1, step):
                janela = df_regime.iloc[inicio : inicio + window_length]
                dados_extraidos.append(
                    window_statistics(
                        janela,
                        condicao=condicao,
                        arquivo=os.path.basename(file_path),
                    )
                )
        except Exception as exc:
            print(f"Aviso: falha ao processar {file_path}: {exc}")

    return pd.DataFrame(dados_extraidos)


def train_knn(train_features, test_features):
    train_groups = train_features["arquivo"]
    X_train = train_features.drop(columns=["condicao", "arquivo"])
    y_train = train_features["condicao"]
    X_test = test_features.drop(columns=["condicao", "arquivo"])
    y_test = test_features["condicao"]

    X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
    X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier()),
        ]
    )

    param_grid = {
        "knn__n_neighbors": [1, 3, 5, 7],
        "knn__weights": ["uniform", "distance"],
        "knn__metric": ["euclidean", "manhattan"],
    }

    cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring="balanced_accuracy",
        n_jobs=-1,
    )

    grid_search.fit(X_train, y_train, groups=train_groups)

    print("\n=== Dataset extraido ===")
    print(f"Janelas de treino: {len(train_features)}")
    print(f"Janelas de teste: {len(test_features)}")
    print(f"Arquivos de treino: {train_groups.nunique()}")
    print(f"Arquivos de teste: {test_features['arquivo'].nunique()}")
    print("Janelas de treino por classe:")
    print(y_train.value_counts())
    print("Janelas de teste por classe:")
    print(y_test.value_counts())

    print("\n=== Melhor KNN ===")
    print("Melhores hiperparametros:", grid_search.best_params_)
    print("Melhor balanced accuracy media:", grid_search.best_score_)

    y_pred = grid_search.predict(X_test)
    y_proba = grid_search.predict_proba(X_test)

    print("\n=== Relatorio no conjunto de teste separado por arquivo ===")
    print(classification_report(y_test, y_pred, zero_division=0))

    labels = grid_search.best_estimator_.classes_

    accuracy = accuracy_score(y_test, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
    precision_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    roc_auc_macro = roc_auc_score(
        y_test,
        y_proba,
        labels=labels,
        multi_class="ovr",
        average="macro",
    )

    print("\n=== Metricas principais ===")
    print(f"Acuracia: {accuracy:.4f}")
    print(f"Balanced accuracy: {balanced_accuracy:.4f}")
    print(f"Precisao macro: {precision_macro:.4f}")
    print(f"Recall macro: {recall_macro:.4f}")
    print(f"F1-score macro: {f1_macro:.4f}")
    print(f"ROC-AUC macro OvR: {roc_auc_macro:.4f}")

    print("\nObservacao sobre desbalanceamento:")
    print(
        "Como as classes possuem quantidades diferentes de janelas, a acuracia "
        "isolada pode esconder erros em classes menores. Por isso, precision, "
        "recall, F1-score e balanced accuracy tambem devem ser analisados."
    )

    cm = confusion_matrix(y_test, y_pred, labels=labels)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.title("Matriz de Confusão - KNN")
    plt.ylabel("Condição real")
    plt.xlabel("Condição prevista")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH)
    plt.savefig(ARTICLE_CONFUSION_MATRIX_PATH, dpi=300)
    print(f"\nMatriz de confusão salva em: {CONFUSION_MATRIX_PATH}")
    print(f"Matriz de confusao do artigo salva em: {ARTICLE_CONFUSION_MATRIX_PATH}")
    plt.show()

    metrics = {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "roc_auc_macro_ovr": roc_auc_macro,
        "best_cv_balanced_accuracy": grid_search.best_score_,
        "train_windows": len(train_features),
        "test_windows": len(test_features),
        "train_original_files": train_groups.nunique(),
        "test_original_files": test_features["arquivo"].nunique(),
    }

    save_training_artifacts(
        df_features=pd.concat([train_features, test_features], ignore_index=True),
        feature_names=X_train.columns,
        metrics=metrics,
        classification_text=classification_report(y_test, y_pred, zero_division=0),
        best_params=grid_search.best_params_,
    )

    return grid_search.best_estimator_


def save_training_artifacts(
    df_features,
    feature_names,
    metrics,
    classification_text,
    best_params,
):
    df_features.to_csv(FEATURES_PATH, index=False)

    with open(FEATURE_NAMES_PATH, "w", encoding="utf-8") as file:
        for name in feature_names:
            file.write(f"{name}\n")

    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(METRICS_CSV_PATH, index=False)

    with open(METRICS_TXT_PATH, "w", encoding="utf-8") as file:
        file.write("=== Melhor KNN ===\n")
        for key, value in best_params.items():
            file.write(f"{key}: {value}\n")

        file.write("\n=== Metricas principais ===\n")
        for key, value in metrics.items():
            if isinstance(value, float):
                file.write(f"{key}: {value:.4f}\n")
            else:
                file.write(f"{key}: {value}\n")

        file.write("\n=== Relatorio de classificacao ===\n")
        file.write(classification_text)

        file.write("\n\n=== Observacao sobre desbalanceamento ===\n")
        file.write(
            "Como as classes possuem quantidades diferentes de janelas, a acuracia "
            "isolada pode esconder erros em classes menores. Por isso, precision, "
            "recall, F1-score, ROC-AUC e balanced accuracy tambem devem ser analisados.\n"
        )

    print(f"Features salvas em: {FEATURES_PATH}")
    print(f"Nomes das features salvos em: {FEATURE_NAMES_PATH}")
    print(f"Metricas salvas em: {METRICS_TXT_PATH}")
    print(f"Metricas em CSV salvas em: {METRICS_CSV_PATH}")


def export_model(model, path=MODEL_PATH):
    joblib.dump(model, path)
    print(f"\nModelo salvo em: {path}")


def load_model(path=MODEL_PATH):
    return joblib.load(path)


if __name__ == "__main__":
    files = collect_files()
    train_files, test_files = train_test_split(
        files,
        test_size=0.2,
        random_state=42,
        stratify=[condicao for _, condicao in files],
    )

    train_features = extract_features_from_files(train_files, window_length=50, step=50)
    test_features = extract_features_from_files(test_files, window_length=50, step=50)

    best_model = train_knn(train_features, test_features)
    export_model(best_model)
