#import library
print('[INFO] Script Untuk Proses Modeling ...')
print('[INFO] Proses Import Library ...')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

from imblearn.over_sampling import SMOTE
import function_library as lib
from function_library import display_importances

from xgboost import XGBClassifier
import lightgbm as lgb
from lightgbm import LGBMRegressor, LGBMClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, precision_recall_curve
from sklearn.metrics import roc_auc_score, log_loss, f1_score, roc_curve, auc
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold

#Import Data
print('[INFO] Proses Import Data ...')

df = pd.read_csv('data_modeling.csv')
print(f'[INFO] Jumlah Data yang Akan di Proses {df.shape[0]} Baris dan {df.shape[1]} Kolom ...')

df['event'] = df['event'].replace({
    0 : 1,
    1 : 0
})

label_positif = df['event'].value_counts()[1]
label_negatif = df['event'].value_counts()[0]

print(f'[INFO] Jumlah Label Positif {label_positif} dan Label Negatif {label_negatif} ...')

#Proses Modeling
print('[INFO] Proses Modeling ...')

x = df.drop(columns=['date', 'machine', 'event'])
y = df[['event']]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, stratify=y, random_state=42)

print('[INFO] Baseline Modeling Logistic Regression ...')
model_log_reg = lib.baseline_model(LogisticRegression(), x_train, y_train, x_test, y_test)
joblib.dump(model_log_reg, 'model_log_reg.pkl')

print('[INFO] Baseline Modeling KNN ...')
model_knn = lib.baseline_model(KNeighborsClassifier(), x_train, y_train, x_test, y_test)
joblib.dump(model_knn, 'model_knn.pkl')

print('[INFO] Baseline Modeling LGBM ...')
model_lgbm = lib.baseline_model(LGBMClassifier(max_depth=4), x_train, y_train, x_test, y_test)
joblib.dump(model_lgbm, 'model_lgbm.pkl')

print('[INFO] Proses Modeling Selesai ...')