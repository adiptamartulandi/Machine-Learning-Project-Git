#import library
print('[INFO] Proses Import Library ...')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

#Import Data
print('[INFO] Proses Import Data Prediction ...')

df = pd.read_csv('data_modeling.csv')
print(f'[INFO] Jumlah Data yang Akan di Proses {df.shape[0]} Baris dan {df.shape[1]} Kolom ...')

df['event'] = df['event'].replace({
    0 : 1,
    1 : 0
})

#Proses Prediction
print('[INFO] Proses Prediction ...')

x = df.drop(columns=['date', 'machine', 'event'])
y = df[['event']]

model_lgbm = joblib.load('model_lgbm.pkl')

hasil_prediksi = model_lgbm.predict_proba(x)[:,1]
hasil_prediksi = np.where(hasil_prediksi < 0.5, 0, 1)

hasil_positif = pd.DataFrame(hasil_prediksi)[0].value_counts()[1]
hasil_negatif = pd.DataFrame(hasil_prediksi)[0].value_counts()[0]

print(f'[INFO] Jumlah Prediksi Label Positif {hasil_positif} dan Label Negatif {hasil_negatif} ...')