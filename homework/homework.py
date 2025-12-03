# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Descompone la matriz de entrada usando PCA. El PCA usa todas las componentes.
# - Estandariza la matriz de entrada.
# - Selecciona las K columnas mas relevantes de la matrix de entrada.
# - Ajusta una maquina de vectores de soporte (svm).
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#

# flake8: noqa: E501
import pickle
import zipfile
from pathlib import Path
import gzip
import json


import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (balanced_accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,)

train_zip = "files/input/train_data.csv.zip"
test_zip = "files/input/test_data.csv.zip"

interno_train = "train_default_of_credit_card_clients.csv"
interno_test = "test_default_of_credit_card_clients.csv"

with zipfile.ZipFile(train_zip, "r") as zf:
    with zf.open(interno_train) as f:
        train_pd = pd.read_csv(f)

with zipfile.ZipFile(test_zip, "r") as zf:
    with zf.open(interno_test) as f:
        test_pd = pd.read_csv(f)

train_pd = train_pd.drop("ID", axis=1)
test_pd = test_pd.drop("ID", axis=1)

train_pd = train_pd.rename(columns={"default payment next month": "default"})
test_pd = test_pd.rename(columns={"default payment next month": "default"})

train_pd = train_pd.dropna()
test_pd = test_pd.dropna()

train_pd = train_pd[(train_pd["EDUCATION"] != 0) & (train_pd["MARRIAGE"] != 0)]
test_pd = test_pd[(test_pd["EDUCATION"] != 0) & (test_pd["MARRIAGE"] != 0)]

train_pd.loc[train_pd["EDUCATION"] > 4, "EDUCATION"] = 4
test_pd.loc[test_pd["EDUCATION"] > 4, "EDUCATION"] = 4


X_train = train_pd.drop(columns=["default"])
y_train = train_pd["default"]

X_test = test_pd.drop(columns=["default"])
y_test = test_pd["default"]


cat_cols = ["SEX", "EDUCATION", "MARRIAGE"]
num_cols = ["LIMIT_BAL","AGE","PAY_0","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6",
            "BILL_AMT1","BILL_AMT2","BILL_AMT3","BILL_AMT4","BILL_AMT5","BILL_AMT6",
            "PAY_AMT1","PAY_AMT2","PAY_AMT3","PAY_AMT4","PAY_AMT5","PAY_AMT6",]

preprocessor = ColumnTransformer(transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),("std", StandardScaler(), num_cols),],remainder="passthrough",)

pipe = Pipeline(steps=[("prep", preprocessor),("pca", PCA()),("kbest", SelectKBest(score_func=f_classif)),("svc", SVC(kernel="rbf", random_state=42)),])

param_grid = {
    "pca__n_components": [20, 21],
    "kbest__k": [12],
    "svc__kernel": ["rbf"],
    "svc__gamma": [0.099],
}

grid = GridSearchCV(estimator=pipe,param_grid=param_grid,cv=10,scoring="balanced_accuracy",refit=True,verbose=1,)

grid.fit(X_train, y_train)

Path("files/models").mkdir(parents=True, exist_ok=True)

with gzip.open("files/models/model.pkl.gz", "wb") as f:
    pickle.dump(grid, f)

y_pred_train = grid.predict(X_train)
y_pred_test = grid.predict(X_test)

train_metrics = {
    "type": "metrics",
    "dataset": "train",
    "precision": precision_score(y_train, y_pred_train),
    "balanced_accuracy": balanced_accuracy_score(y_train, y_pred_train),
    "recall": recall_score(y_train, y_pred_train),
    "f1_score": f1_score(y_train, y_pred_train),
}

test_metrics = {
    "type": "metrics",
    "dataset": "test",
    "precision": precision_score(y_test, y_pred_test),
    "balanced_accuracy": balanced_accuracy_score(y_test, y_pred_test),
    "recall": recall_score(y_test, y_pred_test),
    "f1_score": f1_score(y_test, y_pred_test),
}

tn, fp, fn, tp = confusion_matrix(y_train, y_pred_train).ravel()
cm_train = {
    "type": "cm_matrix",
    "dataset": "train",
    "true_0": {"predicted_0": int(tn), "predicted_1": int(fp)},
    "true_1": {"predicted_0": int(fn), "predicted_1": int(tp)},
}

tn, fp, fn, tp = confusion_matrix(y_test, y_pred_test).ravel()
cm_test = {
    "type": "cm_matrix",
    "dataset": "test",
    "true_0": {"predicted_0": int(tn), "predicted_1": int(fp)},
    "true_1": {"predicted_0": int(fn), "predicted_1": int(tp)},
}

Path("files/output").mkdir(parents=True, exist_ok=True)

with open("files/output/metrics.json", "w", encoding="utf-8") as f:
    f.write(json.dumps(train_metrics) + "\n")
    f.write(json.dumps(test_metrics) + "\n")
    f.write(json.dumps(cm_train) + "\n")
    f.write(json.dumps(cm_test) + "\n")