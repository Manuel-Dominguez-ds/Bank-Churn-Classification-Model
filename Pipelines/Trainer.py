import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score,precision_score,accuracy_score,recall_score,classification_report,confusion_matrix,precision_recall_curve,auc,roc_auc_score,roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
import xgboost as xgb
from sklearn.model_selection import train_test_split,cross_val_score, GridSearchCV, KFold, RandomizedSearchCV,RepeatedStratifiedKFold
from sklearn.preprocessing import OneHotEncoder,StandardScaler,MinMaxScaler,LabelEncoder,Normalizer,MaxAbsScaler
from sklearn.decomposition import PCA
import random
#from imblearn.combine import SMOTEENN
from feature_engine.encoding import CountFrequencyEncoder,OrdinalEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
import lightgbm as lgb
from catboost import CatBoostClassifier
import pickle
from datetime import date
import datetime
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from Functions import *
from random import randint
#import tensorflow as tf


'''
Build a training pipeline (training.py) where you will read this table, 
prepare it for modelling, run a hyperparameters search, store the best
trained model in any desired format, and store a text or pdf file with 
some evaluation metrics to describe the model.
'''


class Trainer():

    def __init__(self,df_path) -> None:
        print("\n > Initialized Training Pipeline.")
        self.df_path = df_path

    def orchestrator(self):
        try:
            print("\n > Starting Training process.")
            
            dataset = self.read_data(path=self.df_path)
            dataset=dataset.dropna()
            X_train, X_val, y_train, y_val = self.train_validation_split(dataset)
            X_train, X_val,y_train,y_val = self.preprocessing(X_train, X_val, y_train, y_val)
            
            self.X_train = X_train
            self.X_val = X_val
            self.y_train = y_train
            self.y_val = y_val
            self.seed = randint(0,1000)
            
            self.train_models()
            self.evaluate_and_save_results()
            print("\n > Finished Training process succesfully")
        except Exception as e:
            print(f"\n > Exception during the training process: {e}")
            #traceback.print_exc()
    
    def read_data(self,path):
        print("\n > Reading data.")
        df=pd.read_csv(path)
        return df
    
    def preprocessing(self, X_train, X_val,y_train,y_val):
        print("\n > Preprocessing data.")
        print("\n > Null treatment.")
        # X_train=X_train.dropna()
        # y_train=y_train.dropna()
        # X_val=X_val.dropna()
        # y_val=y_val.dropna()
        
        num_imputer=SimpleImputer(strategy='mean')
        cat_imputer=SimpleImputer(strategy='most_frequent')
        X_train=imputer(X_train,cat_imputer,num_imputer,train=True)
        
        print("\n > Performing Feature Engineering.")
        X_train=X_train.drop(['Surname','Passport','Gender'],axis=1)
        X_val=X_val.drop(['Surname','Passport','Gender'],axis=1)
        X_train=new_variables(X_train)
        X_val=new_variables(X_val)
        
        #print("\n > Encoding data.")
        print(X_train.shape,X_val.shape,y_train.shape,y_val.shape)
        X_train,X_val=encoder(X_train,X_val,y_train)
        
        
        #print("\n > Scaling Data.")
        X_train,X_val=scaler(X_train,X_val)
        
        print("\n > Preprocessed Data.")
        return X_train, X_val, y_train, y_val

    def train_validation_split(self, df):
        y = df.Exited
        X = df.drop(['Exited','Id'], axis=1	)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3,random_state=42)
        splits = X_train, X_val, y_train, y_val
        print(f"\n > Performed an 75/25 split. Training set has {X_train.shape[0]} examples. Validation set has {X_val.shape[0]} examples.")
        return splits

    def train_models(self):
        print("\n > Training models. Performing HyperParameter tunning with Gridsearch cross validation. This could take a while..🧠")
        # TODO merge al metrics in a single dictionary
        self.metrics={
            'Mean Squared Error':0,
            'Mean Absolut Error':0,
            'Root Mean Squared Deviation':0,
            'R2':0
        }
        self.precision = []
        self.recall = []
        self.accuracy = []
        self.f1 = []
        self.auc=[]
        self.prauc=[]
        self.final_score=[]
        self.model_instances = {}
        self.models = {'Models':[],'Accuracy':[],'Precision':[],'Recall':[],'F1':[],'AUC':[],'PRAUC':[],'FinalScore':[]}
        
        skf=StratifiedKFold(n_splits=3,shuffle=True)
        #self.train_logistic_regression(skf)
        #self.train_random_forest_classifier(skf)
        #self.train_svm(skf)
        self.train_lightgbm(skf)
        self.train_xgboost(skf)
        self.train_catboost(skf)
        self.train_gradient_boosting(skf)
        self.train_voting_classifier(skf)
        
    def train_logistic_regression(self,skf):
        X_train = self.X_train
        y_train = self.y_train
        X_val = self.X_val
        y_val = self.y_val
        model = LogisticRegression()
        gscv = GridSearchCV(
            model,
            param_grid={},
            cv=skf,
            n_jobs=-1
        )
        gscv.fit(X_train, y_train)

        best_estimator = gscv.best_estimator_

        y_pred = best_estimator.predict_proba(X_val)[:, 1]
        y_pred = (y_pred > 0.49).astype(int)
        
        precision, recall, thresholds = precision_recall_curve(y_val, y_pred)
        pr_auc = auc(recall, precision)
        auc_score = roc_auc_score(y_val, y_pred)
        accuracy = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        precision=precision_score(y_val,y_pred)
        recall=recall_score(y_val,y_pred)
        self.accuracy.append(accuracy)
        self.f1.append(f1)
        self.precision.append(precision)
        self.recall.append(recall)
        self.models['Models'].append('Logistic Regression')
        self.models['Accuracy'].append(accuracy)
        self.models['Precision'].append(precision)
        self.models['Recall'].append(recall)
        self.models['F1'].append(f1)
        self.models['AUC'].append(auc_score)
        self.models['PRAUC'].append(pr_auc)
        self.models['FinalScore'].append((accuracy*1.5+f1*1.5+precision*0.75+recall*.75+auc_score*1.2+pr_auc*1.2)/6)
        self.model_instances.setdefault('Logistic Regression',best_estimator)
  
        print("\n > Results for Logistic Regression\n")
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {round(precision)}")
        print(f"F1 Score: {round(f1,2)}")
        print(f"Recall: {round(recall,2)}")
        print(f"AUC: {round(auc_score,2)}")
        print(f"PR AUC: {round(pr_auc,2)}")

    def train_random_forest_classifier(self, skf):
        X_train = self.X_train
        y_train = self.y_train
        X_val = self.X_val
        y_val = self.y_val

        param_grid = {
            'n_estimators': [100, 200, 300, 400, 500],
            'max_depth': [10, 20, 30, 40, 50],
            'max_features': ['auto', 'sqrt'],
            'min_samples_leaf': [1, 2, 4],
            'min_samples_split': [2, 5, 10],
            'bootstrap': [True, False]
        }
        
        model = RandomForestClassifier()
        gscv = GridSearchCV(
            model,
            param_grid=param_grid,
            cv=skf,
            n_jobs=-1
        )
        gscv.fit(X_train, y_train)

        best_estimator = gscv.best_estimator_

        y_pred = best_estimator.predict_proba(X_val)[:, 1]
        y_pred = (y_pred > 0.49).astype(int)

        precision, recall, thresholds = precision_recall_curve(y_val, y_pred)
        pr_auc = auc(recall, precision)
        auc_score = roc_auc_score(y_val, y_pred)
        accuracy = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)
        self.accuracy.append(accuracy)
        self.f1.append(f1)
        self.precision.append(precision)
        self.recall.append(recall)
        self.models['Models'].append('Random Forest Classifier')
        self.models['Accuracy'].append(accuracy)
        self.models['Precision'].append(precision)
        self.models['Recall'].append(recall)
        self.models['F1'].append(f1)
        self.models['AUC'].append(auc_score)
        self.models['PRAUC'].append(pr_auc)
        self.models['FinalScore'].append((accuracy*1.5+f1*1.5+precision*0.75+recall*.75+auc_score*1.2+pr_auc*1.2)/6)
        self.model_instances.setdefault('Random Forest Classifier', best_estimator)

        print("\n > Results for Random Forest Classifier\n")
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {round(precision)}")
        print(f"F1 Score: {round(f1, 2)}")
        print(f"Recall: {round(recall, 2)}")
        print(f"AUC: {round(auc_score, 2)}")
        print(f"PR AUC: {round(pr_auc, 2)}")

    def train_svm(self, skf):
        X_train = self.X_train
        y_train = self.y_train
        X_val = self.X_val
        y_val = self.y_val
        model = SVC()
        
        param_grid={
            # 'C':[0.1,1,10,100],
            # 'gamma':[1,0.1,0.01,0.001],
            # 'kernel':['rbf']
        }
        
        gscv = GridSearchCV(
            model,
            param_grid=param_grid,
            cv=skf,
            n_jobs=-1
        )
        gscv.fit(X_train, y_train)

        best_estimator = gscv.best_estimator_

        y_pred = best_estimator.predict(X_val)

        precision, recall, thresholds = precision_recall_curve(y_val, y_pred)
        pr_auc = auc(recall, precision)
        auc_score = roc_auc_score(y_val, y_pred)
        accuracy = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)
        self.accuracy.append(accuracy)
        self.f1.append(f1)
        self.precision.append(precision)
        self.recall.append(recall)
        self.models['Models'].append('SVM')
        self.models['Accuracy'].append(accuracy)
        self.models['Precision'].append(precision)
        self.models['Recall'].append(recall)
        self.models['F1'].append(f1)
        self.models['AUC'].append(auc_score)
        self.models['PRAUC'].append(pr_auc)
        self.models['FinalScore'].append((accuracy*1.5+f1*1.5+precision*0.75+recall*.75+auc_score*1.2+pr_auc*1.2)/6)
        self.model_instances.setdefault('SVM', best_estimator)

        print("\n > Results for SVM\n")
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {round(precision)}")
        print(f"F1 Score: {round(f1, 2)}")
        print(f"Recall: {round(recall, 2)}")
        print(f"AUC: {round(auc_score, 2)}")
        print(f"PR AUC: {round(pr_auc, 2)}")

    def train_lightgbm(self, skf):
        X_train = self.X_train
        y_train = self.y_train
        X_val = self.X_val
        y_val = self.y_val
        
        param_grid = {
        'boosting_type':['dart','gbdt'],
        'num_leaves': [20,30,40],
        'learning_rate': [0.1,0.01,0.001],
        'n_estimators': [100,200,500],
        'max_depth': [10,20,30]}
        
        model = lgb.LGBMClassifier(       
                                boosting_type='dart',
                                num_leaves=20,
                                learning_rate=0.1,
                                max_depth=10,
                                n_estimators=100,
                                objective='binary',
                                metric='binary_logloss',
                                feature_fraction=0.8,
                                bagging_fraction=0.6,
                                bagging_freq=10,
                                lambda_l1=1,
                                lambda_l2=1,
                                min_data_in_leaf=20,
                                min_gain_to_split=0.1)
        gscv = GridSearchCV(
            model,
            param_grid=param_grid,
            cv=skf,
            n_jobs=-1
        )
        gscv.fit(X_train, y_train)

        best_estimator = gscv.best_estimator_

        y_pred = best_estimator.predict_proba(X_val)[:, 1]
        y_pred = (y_pred > 0.49).astype(int)

        precision, recall, thresholds = precision_recall_curve(y_val, y_pred)
        pr_auc = auc(recall, precision)
        auc_score = roc_auc_score(y_val, y_pred)
        accuracy = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)
        self.accuracy.append(accuracy)
        self.f1.append(f1)
        self.precision.append(precision)
        self.recall.append(recall)
        self.models['Models'].append('LightGBM')
        self.models['Accuracy'].append(accuracy)
        self.models['Precision'].append(precision)
        self.models['Recall'].append(recall)
        self.models['F1'].append(f1)
        self.models['AUC'].append(auc_score)
        self.models['PRAUC'].append(pr_auc)
        self.models['FinalScore'].append((accuracy*1.5+f1*1.5+precision*0.75+recall*.75+auc_score*1.2+pr_auc*1.2)/6)
        self.model_instances.setdefault('LightGBM', best_estimator)

        print("\n > Results for LightGBM\n")
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {round(precision)}")
        print(f"F1 Score: {round(f1, 2)}")
        print(f"Recall: {round(recall, 2)}")
        print(f"AUC: {round(auc_score, 2)}")
        print(f"PR AUC: {round(pr_auc, 2)}")

    def train_xgboost(self, skf):
        X_train = self.X_train
        y_train = self.y_train
        X_val = self.X_val
        y_val = self.y_val
        param_grid={
            'n_estimators': [50, 100, 150],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 1.0]
        }
        
        model = xgb.XGBClassifier()
        gscv = GridSearchCV(
            model,
            param_grid=param_grid,
            cv=skf,
            n_jobs=-1
        )
        gscv.fit(X_train, y_train)

        best_estimator = gscv.best_estimator_

        y_pred = best_estimator.predict_proba(X_val)[:, 1]
        y_pred = (y_pred > 0.49).astype(int)

        precision, recall, thresholds = precision_recall_curve(y_val, y_pred)
        pr_auc = auc(recall, precision)
        auc_score = roc_auc_score(y_val, y_pred)
        accuracy = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)
        self.accuracy.append(accuracy)
        self.f1.append(f1)
        self.precision.append(precision)
        self.recall.append(recall)
        self.models['Models'].append('XGBoost')
        self.models['Accuracy'].append(accuracy)
        self.models['Precision'].append(precision)
        self.models['Recall'].append(recall)
        self.models['F1'].append(f1)
        self.models['AUC'].append(auc_score)
        self.models['PRAUC'].append(pr_auc)
        self.models['FinalScore'].append((accuracy*1.5+f1*1.5+precision*0.75+recall*.75+auc_score*1.2+pr_auc*1.2)/6)
        self.model_instances.setdefault('XGBoost', best_estimator)

        print("\n > Results for XGBoost\n")
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {round(precision)}")
        print(f"F1 Score: {round(f1, 2)}")
        print(f"Recall: {round(recall, 2)}")
        print(f"AUC: {round(auc_score, 2)}")
        print(f"PR AUC: {round(pr_auc, 2)}")

    def train_catboost(self, skf):
        X_train = self.X_train
        y_train = self.y_train
        X_val = self.X_val
        y_val = self.y_val
        
        param_grid = {
            'iterations': [50, 100, 150],
            'depth': [4, 6, 10],
            'learning_rate': [0.01, 0.1, 0.2]
        }

        model = CatBoostClassifier(verbose=False)
        gscv = GridSearchCV(
            model,
            param_grid=param_grid,
            cv=skf,
            n_jobs=-1
        )
        gscv.fit(X_train, y_train)

        best_estimator = gscv.best_estimator_

        y_pred = best_estimator.predict_proba(X_val)[:, 1]
        y_pred = (y_pred > 0.49).astype(int)

        precision, recall, thresholds = precision_recall_curve(y_val, y_pred)
        pr_auc = auc(recall, precision)
        auc_score = roc_auc_score(y_val, y_pred)
        accuracy = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)
        self.accuracy.append(accuracy)
        self.f1.append(f1)
        self.precision.append(precision)
        self.recall.append(recall)
        self.models['Models'].append('CatBoost')
        self.models['Accuracy'].append(accuracy)
        self.models['Precision'].append(precision)
        self.models['Recall'].append(recall)
        self.models['F1'].append(f1)
        self.models['AUC'].append(auc_score)
        self.models['PRAUC'].append(pr_auc)
        self.models['FinalScore'].append((accuracy*1.5+f1*1.5+precision*0.75+recall*.75+auc_score*1.2+pr_auc*1.2)/6)
        self.model_instances.setdefault('CatBoost', best_estimator)

        print("\n > Results for CatBoost\n")
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {round(precision)}")
        print(f"F1 Score: {round(f1, 2)}")
        print(f"Recall: {round(recall, 2)}")
        print(f"AUC: {round(auc_score, 2)}")
        print(f"PR AUC: {round(pr_auc, 2)}")

    def train_gradient_boosting(self, skf):
        X_train = self.X_train
        y_train = self.y_train
        X_val = self.X_val
        y_val = self.y_val
        
        param_grid = {
            'n_estimators': [50, 100, 150],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2]
        }
        model = GradientBoostingClassifier()
        gscv = GridSearchCV(
            model,
            param_grid=param_grid,
            cv=skf,
            n_jobs=-1
        )
        gscv.fit(X_train, y_train)

        best_estimator = gscv.best_estimator_

        y_pred = best_estimator.predict_proba(X_val)[:, 1]
        y_pred = (y_pred > 0.49).astype(int)

        precision, recall, thresholds = precision_recall_curve(y_val, y_pred)
        pr_auc = auc(recall, precision)
        auc_score = roc_auc_score(y_val, y_pred)
        accuracy = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)
        self.accuracy.append(accuracy)
        self.f1.append(f1)
        self.precision.append(precision)
        self.recall.append(recall)
        self.models['Models'].append('Gradient Boosting')
        self.models['Accuracy'].append(accuracy)
        self.models['Precision'].append(precision)
        self.models['Recall'].append(recall)
        self.models['F1'].append(f1)
        self.models['AUC'].append(auc_score)
        self.models['PRAUC'].append(pr_auc)
        self.models['FinalScore'].append((accuracy*1.5+f1*1.5+precision*0.75+recall*.75+auc_score*1.2+pr_auc*1.2)/6)
        self.model_instances.setdefault('Gradient Boosting', best_estimator)

        print("\n > Results for Gradient Boosting\n")
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {round(precision)}")
        print(f"F1 Score: {round(f1, 2)}")
        print(f"Recall: {round(recall, 2)}")
        print(f"AUC: {round(auc_score, 2)}")
        print(f"PR AUC: {round(pr_auc, 2)}")

    def train_voting_classifier(self, skf):
        X_train = self.X_train
        y_train = self.y_train
        X_val = self.X_val
        y_val = self.y_val
        
        model1 = self.model_instances['Logistic Regression']
        model2 = self.model_instances['Random Forest Classifier']
        model3 = self.model_instances['LightGBM']
        model4 = self.model_instances['XGBoost']
        model5 = self.model_instances['CatBoost']

        model = VotingClassifier(
            estimators=[('lr', model1), ('rf', model2), ('lgbm', model3), ('xgb', model4), ('catboost', model5)],
            voting='soft'
        )

        gscv = GridSearchCV(
            model,
            param_grid={},
            cv=skf,
            n_jobs=-1
        )
        gscv.fit(X_train, y_train)

        best_estimator = gscv.best_estimator_

        y_pred = best_estimator.predict_proba(X_val)[:, 1]
        y_pred = (y_pred > 0.49).astype(int)

        precision, recall, thresholds = precision_recall_curve(y_val, y_pred)
        pr_auc = auc(recall, precision)
        auc_score = roc_auc_score(y_val, y_pred)
        accuracy = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)
        self.accuracy.append(accuracy)
        self.f1.append(f1)
        self.precision.append(precision)
        self.recall.append(recall)
        self.models['Models'].append('Voting Classifier')
        self.models['Accuracy'].append(accuracy)
        self.models['Precision'].append(precision)
        self.models['Recall'].append(recall)
        self.models['F1'].append(f1)
        self.models['AUC'].append(auc_score)
        self.models['PRAUC'].append(pr_auc)
        self.models['FinalScore'].append((accuracy*1.5+f1*1.5+precision*0.75+recall*.75+auc_score*1.2+pr_auc*1.2)/6)
        self.model_instances.setdefault('Voting Classifier', best_estimator)

        print("\n > Results for Voting Classifier\n")
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {round(precision)}")
        print(f"F1 Score: {round(f1, 2)}")
        print(f"Recall: {round(recall, 2)}")
        print(f"AUC: {round(auc_score, 2)}")
        print(f"PR AUC: {round(pr_auc, 2)}")

    def evaluate_and_save_results(self):
        print(f"\n > Evaluating and saving best model.")
        models_results=pd.DataFrame(self.models)
        # save results to txt
        filename = f'training_output - {date.today()}.csv'
        with open(filename, 'a') as file:
            file.write("Models, Accuracy, Precision, Recall, F1, AUC, PRAUC, FinalScore\n")
            for i in range(models_results.shape[0]):
                line = str(models_results.Models[i]) + "," + str(models_results.Accuracy[i])+","+ str(models_results.Precision[i])+ "," + str(models_results.Recall[i]) + "," + str(models_results.F1[i]) + "," + str(models_results.AUC[i]) + "," + str(models_results.PRAUC[i])+ "," + str(models_results.FinalScore[i])
                file.write(line + "\n")
        print(f"Training results have been saved to {filename}")

        best_model = self.get_best_model()
        self.save_model(best_model)

    def get_best_model(self):
        # get best model based on the RMSE Metric
        models_results=pd.DataFrame(self.models)
        best_model_name=models_results.loc[models_results['FinalScore'].idxmax()]['Models']
        print(f"\n ----> 🧠 Best model is {best_model_name} 🧠 <----")
        return self.model_instances[best_model_name]

    def save_model(self, best):
        # Dump model as pkl file
        with open('Pickle/best_model.pkl', 'wb') as file:
            print(f"\n > Exporting best model to pkl file 'best_model.pkl'⬇️💾")
            pickle.dump(best, file)
            