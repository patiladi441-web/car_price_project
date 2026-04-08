import pandas as pd
import seaborn as sb
import matplotlib.pyplot as mp
import streamlit as st
d=pd.read_csv('/Users/adinathpatil/Downloads/Car details v3.csv')

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer,KNNImputer

ohe=OneHotEncoder(drop='first',handle_unknown='ignore')

d['torque_nm'] = pd.to_numeric(d['torque'].str.extract('(\d+\.?\d*)')[0], errors='coerce')
d['torque_rpm'] = pd.to_numeric(d['torque'].str.extract('@\s*(\d+)')[0], errors='coerce')
d.drop(columns=['torque'], inplace=True)

d['name'] = d['name'].apply(lambda x: x.split()[0])
d['mileage'] = d['mileage'].str.split().str[0]
d['mileage'] = pd.to_numeric(d['mileage'], errors='coerce')
d['engine'] = d['engine'].str.split().str[0]
d['engine'] = pd.to_numeric(d['engine'], errors='coerce')
d['max_power'] = d['max_power'].str.split().str[0]
d['max_power'] = pd.to_numeric(d['max_power'], errors='coerce')

d['car_age'] = 2024 - d['year']
d.drop(columns=['year'], inplace=True)

mileag=Pipeline(
   [ ('imp',KNNImputer(n_neighbors=5))]
)

ct=ColumnTransformer(
    transformers=[
        ('name',ohe,['name']),
        ('fuel',ohe,['fuel']),
        ('owner',ohe,['owner']),
        ('trans',ohe,['transmission']),
        ('seltype',ohe,['seller_type']),
        ('milg',mileag,['mileage']),
        ('eng',KNNImputer(n_neighbors=5),['engine']),
        ('power',KNNImputer(n_neighbors=5),['max_power']),
        ('seats',SimpleImputer(strategy='median'),['seats']),
        ('torque_nm',KNNImputer(n_neighbors=5),['torque_nm']),
        ('torque_rpm',KNNImputer(n_neighbors=5),['torque_rpm']),
        
        

    ],remainder='passthrough'
)

x=d.drop(columns=['selling_price'])
import numpy as np
y = np.log1p(d['selling_price'])



y=d['selling_price']

from sklearn.model_selection import train_test_split
xtr,xte,ytr,yte=train_test_split(x,y,test_size=.2,random_state=42)
xtrt=ct.fit_transform(xtr)
xtet=ct.transform(xte)

import xgboost as xgb
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV



xg=xgb.XGBRegressor(
    subsample= 0.7, n_estimators= 600, min_child_weight= 1, max_depth =6, learning_rate = 0.1, gamma = 0
)


xg.fit(xtrt,ytr)
yp=xg.predict(xtet)
print(r2_score(yte,yp))
print('cr',cross_val_score(xg,xtrt,ytr,scoring='r2',cv=5).mean())

# param={
#     'n_estimators':[200,400,600],
#     'max_depth':[4,6,8],
#     'learning_rate':[.1,.01,.001],
#     'subsample': [0.7, 0.8, 1.0],
#     'min_child_weight': [1, 3, 5],
#     'gamma': [0, 0.1, 0.3]
# }
# search=RandomizedSearchCV(
#     estimator=xg,
#     param_distributions=param,
#     verbose=1,
#     n_jobs=-1,
#     scoring='r2',
#     n_iter=30,
#     random_state=42

# )
# search.fit(xtrt, ytr)
# yp=search.predict(xtet)
# print("Best params:", search.best_params_)
# best_model = search.best_estimator_
# print(search.best_score_)
