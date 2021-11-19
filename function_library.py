#Import Library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
from imblearn.over_sampling import SMOTE

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

#List Function

#function for plot correlation
def plot_correlation(df):
    f = plt.figure() 
    f.set_figwidth(12) 
    f.set_figheight(9) 
    sns.heatmap(df.corr(), annot = True, vmin=-1, vmax=1, center= 0, cmap= 'Blues', linewidths=1, linecolor='black')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.show()

#function for plot features importances
def feature_importances_lgbm(model):
    lgb.plot_importance(model, height=0.5, max_num_features=10,
                            grid=False, importance_type='gain', precision=0,
                            title='Feature Importances Based on Information Gain',
                            xlabel='Score')

#function for plot confusin matrix
def confusion_matrix_plot(y, y_prediksi):
    cm_train = confusion_matrix(y, y_prediksi)
    group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
    group_counts = ['{0:0.0f}'.format(value) for value in cm_train.flatten()]
    group_percentages = ['{0:.2%}'.format(value) for value in cm_train.flatten()/np.sum(cm_train)]
    labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    sns.heatmap(cm_train, annot=labels, fmt='', cmap='Blues')

#function for plot precision and recall
def plot_precision_and_recall(y, y_prediksi):
    precision, recall, threshold = precision_recall_curve(y, y_prediksi)
    plt.figure(figsize=(7, 3))
    plt.plot(threshold, precision[:-1], "r-", label="precision", linewidth=5)
    plt.plot(threshold, recall[:-1], "b", label="recall", linewidth=5)
    plt.xlabel("threshold", fontsize=19)
    plt.legend(loc="upper right", fontsize=19)
    plt.ylim([0, 1])
    plt.title('LGBM Precision vs Recall')

#function for plot ROC-AUC
def auc_curve(y,prob):
    fpr,tpr,threshold = roc_curve(y,prob)
    roc_auc = auc(fpr,tpr)
 
    plt.figure(figsize=(8, 4))
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc="lower right")
 
    plt.show()
    
    return fpr, tpr

#function for training LGBM Model with 10-fold CV
# Membuat model Kfold LightGBM
def kfold_lightgbm(df, num_folds, stratified=False, categorical=None):
    global subs,train_df,test_df
    
    train_df = df.drop(columns=['date', 'machine'])
    print("Starting LightGBM. Data shape: {}".format(train_df.shape))

    # Cross validation model
    if stratified:
        folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=2020)
    else:
        folds = KFold(n_splits= num_folds, shuffle=True, random_state=1997)
        
    # Create arrays and dataframes to store results
    oof_preds = np.zeros(train_df.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = [f for f in train_df.columns if f not in ['event']]
    auc=[]
    
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['event'])):
        train_x, train_y = train_df[feats].iloc[train_idx], train_df['event'].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['event'].iloc[valid_idx]

        # LightGBM parameters
        clf = LGBMClassifier(
                application= 'binary',
                objective= 'binary',
                metric= 'auc',
                boosting= 'gbdt',
                num_leaves= 2**5,
                subsample= 0.8,  # Subsample ratio of the training instance.
                subsample_freq= 1,  # frequence of subsample, <=0 means no enable
                feature_fraction= 0.8,
                feature_fraction_bynode= 0.8,
                learning_rate= 0.075,
                max_depth = 5,
                verbose= 20,
                min_child_weight=0,
                min_child_samples= 100,
                max_bin=100,
                is_unbalance=True,
                num_threads=1,
                n_estimators=1000,
                lambda_l1=5,
                lambda_l2=5,
                min_data_in_leaf= 20,
                min_gain_to_split= 0.01
        )


        clf.fit(train_x, train_y, eval_set=[(valid_x, valid_y)], 
            eval_metric= 'auc', verbose= 200, early_stopping_rounds= 200,categorical_feature=categorical)

        oof_preds[valid_idx] = clf.predict_proba(valid_x, num_iteration=clf.best_iteration_)[:, 1]

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx])))
        auc.append(roc_auc_score(valid_y, oof_preds[valid_idx]))

    print('Full AUC score %.6f' % roc_auc_score(train_df['event'], oof_preds))
    print(f'Average AUC score : {round(np.mean(auc),6)} +- {round(np.std(auc),6)}')
    
    display_importances(feature_importance_df)
    return feature_importance_df, oof_preds

#function for running baseline model without tuning
def baseline_model(model, x_train, y_train, x_test, y_test):
    model_fit = model
    model_fit.fit(x_train, y_train)
    y_pred_proba = model_fit.predict_proba(x_test)[:,1]
    y_pred_proba_train = model_fit.predict_proba(x_train)[:,1]
    print('[INFO] Test roc_auc_score is {:.5}'.format(roc_auc_score(y_test, y_pred_proba)))

    return model_fit
    
# Display/plot feature importance untuk model LGBM
def display_importances(feature_importance_df_):
    global best_features
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:40].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    plt.figure(figsize=(8, 10))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig('lgbm_importances01.png')

#END