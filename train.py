"""
Module to train a machine learning model.
"""

import warnings
import datetime
import joblib
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

warnings.filterwarnings('ignore')

data = pd.read_csv('car data.xls')

date_time = datetime.datetime.now()
data['Age'] = date_time.year - data['Year']
data.drop('Year', axis=1, inplace=True)
data = data[~((data['Selling_Price'] >= 33.0) & (data['Selling_Price'] <= 35.0))]
data['Fuel_Type'] = data['Fuel_Type'].map({'Petrol': 0, 'Diesel': 1, 'CNG': 2})
data['Seller_Type'] = data['Seller_Type'].map({'Dealer': 0, 'Individual': 1})
data['Transmission'] = data['Transmission'].map({'Manual': 0, 'Automatic': 1})

X = data.drop(['Car_Name', 'Selling_Price'], axis=1)
y = data['Selling_Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

lr = LinearRegression()
lr.fit(X_train, y_train)

rf = RandomForestRegressor()
rf.fit(X_train, y_train)

xgb = GradientBoostingRegressor()
xgb.fit(X_train, y_train)

xg = XGBRegressor()
xg.fit(X_train, y_train)

y_pred1 = lr.predict(X_test)
y_pred2 = rf.predict(X_test)
y_pred3 = xgb.predict(X_test)
y_pred4 = xg.predict(X_test)

score1 = metrics.r2_score(y_test, y_pred1)
score2 = metrics.r2_score(y_test, y_pred2)
score3 = metrics.r2_score(y_test, y_pred3)
score4 = metrics.r2_score(y_test, y_pred4)

final_data = pd.DataFrame({'Models': ['LR', 'RF', 'GBR', 'XG'],
                           "R2_SCORE": [score1, score2, score3, score4]})

xg_final = XGBRegressor()
xg_final.fit(X, y)
joblib.dump(xg_final, 'car_price_predictor.pkl')
model = joblib.load('car_price_predictor.pkl')

print(final_data)
