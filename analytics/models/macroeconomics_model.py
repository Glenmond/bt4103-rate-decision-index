import pandas as pd
import numpy as np
import math
from scipy.stats.mstats import gmean, hmean

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.tools.tools import add_constant

import warnings 
warnings.filterwarnings('ignore')

from .base_model import Model

class MacroModel(Model):
    def __init__(self, data):
        self.data = data

        # Initialise train and test sets as None first
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None

        # Initialize model results as None first
        self.fitted_model = None
        self.model_details = None

        # Initialize test and val predictions as None first
        self.validation_results = None
        self.testing_results = None

    def preprocess(self):
        # Preprocess the data obtained from the API call

        # Add the HD index
        hd = pd.read_csv('./models/final_df.csv')
        hd.index = hd['Date']
        hd.index = pd.to_datetime(hd.index).to_period('M')
        hd = hd.drop("Date", axis=1)
        # hd = hd.shift(1) # Last month's sentiments affect this month's rate
        hd = hd[hd.index >= '2003-01']
        hd = hd[hd.index <= '2021-04']

        # Combine the different sentiments by using harmonic mean
        hd['Score_Statement'] = MinMaxScaler().fit_transform(hd['Score_Statement'].values.reshape(-1,1))
        hd['Score_Minutes'] = MinMaxScaler().fit_transform(hd['Score_Minutes'].values.reshape(-1,1))
        hd['Score_News'] = MinMaxScaler().fit_transform(hd['Score_News'].values.reshape(-1,1))
        hd['Overall'] = hmean([hd['Score_Statement'],hd['Score_Minutes'],hd['Score_News']])
        
        # Scale then use arithmetic mean
        # hd['Score_Statement'] = hd['Score_Statement'].apply(lambda x: x/(hd['Score_Statement'].max()) if x > 0 else -(x/(hd['Score_Statement'].min())))
        # hd['Score_Minutes'] = hd['Score_Minutes'].apply(lambda x: x/(hd['Score_Minutes'].max()) if x > 0 else -(x/(hd['Score_Minutes'].min())))
        # hd['Score_News'] = hd['Score_News'].apply(lambda x: x/(hd['Score_News'].max()) if x > 0 else -(x/(hd['Score_News'].min())))
        # hd['Overall'] = [np.mean([x,y,z]) for x,y,z in zip(hd['Score_Statement'], hd['Score_Minutes'], hd['Score_News'])]

        # Add the HD index to the data
        self.data['HD_index'] = hd['Overall']

        # Get the 1 period ago rate decision
        self.data['shifted_target'] = self.data['target'].shift(1)
        self.data.dropna(inplace=True)

        # Add interactions
        self.data['MEDCPI_PPIACO'] = self.data['MEDCPI'] * self.data['PPIACO']
        self.data['EMRATIO_MEDWAGES'] = self.data['EMRATIO'] * self.data['MEDWAGES']

        # Do train test split
        X = self.data.copy().drop('target', axis=1)
        y = pd.DataFrame(self.data['target'])
        test_proportion = int(0.1*len(self.data))
        X_train = X[:-test_proportion]
        y_train = y[:-test_proportion]
        self.X_test = X[-test_proportion:]
        self.y_test = y[-test_proportion:]
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=42)

        # Do some transformations
        self.X_train['HD_index'] = np.power(self.X_train['HD_index'],1/2)
        self.X_test['HD_index'] = np.power(self.X_test['HD_index'],1/2)
        self.X_val['HD_index'] = np.power(self.X_val['HD_index'],1/2)

        # Do some scalings
        scaler = StandardScaler()
        scaler.fit(self.X_train) # fit the scaler only on training set
        self.X_train = pd.DataFrame(scaler.transform(self.X_train), columns=self.X_train.columns, index=self.X_train.index)
        self.X_val = pd.DataFrame(scaler.transform(self.X_val), columns=self.X_val.columns, index=self.X_val.index)
        self.X_test = pd.DataFrame(scaler.transform(self.X_test), columns=self.X_test.columns, index=self.X_test.index)

        # Rearrange the input features
        self.X_train = self.X_train[['T10Y3M', 'EMRATIO_MEDWAGES','EMRATIO', 'GDPC1','MEDCPI','MEDCPI_PPIACO','HD_index','shifted_target']]
        self.X_val = self.X_val[['T10Y3M', 'EMRATIO_MEDWAGES','EMRATIO', 'GDPC1','MEDCPI','MEDCPI_PPIACO','HD_index','shifted_target']]
        self.X_test = self.X_test[['T10Y3M', 'EMRATIO_MEDWAGES','EMRATIO', 'GDPC1','MEDCPI','MEDCPI_PPIACO','HD_index','shifted_target']]

    def fit_data(self):
        exog = self.X_train
        endog = self.y_train

        try:
            # add a constant term for the regression
            exog = sm.add_constant(exog) 
            # this is the same as OLS
            model = sm.GLM(endog,exog, family = sm.families.Gaussian(sm.families.links.identity()))
            model_results = model.fit()
            self.fitted_model = model_results
            self.model_details = model_results.summary()
        except (TypeError):
            print("The dataset is not defined properly, make sure you preprocess the data first before fitting!") 

    def predict(self, data):
        # TODO add in the Nonetype checks for the date (X_test or X_val)
        # TODO make sure data can be anything list-like, can be a normal python list, or np.array or pd.series etc
        
        y_pred_res = self.fitted_model.get_prediction(sm.add_constant(data))
        y_pred = y_pred_res.predicted_mean
        y_pred = np.array([[x] for x in y_pred])
        y_pred = [np.array([0]) if x < 0 else x for x in y_pred]
        
        return y_pred_res, y_pred

    def assess_val_set_performance(self):
        y_pred_res, y_pred = self.predict(self.X_val)
        self.validation_results = y_pred_res 
        
        # TODO: add in the NoneType checks for y_val
        r2 = r2_score(self.y_val, y_pred)
        adj_r2 = 1-(1-r2)*(len(self.X_val)-1)/(len(self.X_val)-4-1)
        rmse = math.sqrt(mean_squared_error(self.y_val, y_pred))

        print("Performance on Validation Set")
        print(f"\tThe R2 score is {r2}")
        print(f"\tThe Adjusted R2 score is {adj_r2}")
        print(f"\tThe RMSE score is {rmse}")
        
    def assess_test_set_performance(self):
        y_pred_res, y_pred = self.predict(self.X_test)
        self.testing_result = y_pred_res
        
        # TODO: add in the NoneType checks for y_test
        r2 = r2_score(self.y_test, y_pred)
        adj_r2 = 1-(1-r2)*(len(self.X_test)-1)/(len(self.X_test)-4-1)
        rmse = math.sqrt(mean_squared_error(self.y_test, y_pred))

        print("Performance on Test Set")
        print(f"\tThe R2 score is {r2}")
        print(f"\tThe Adjusted R2 score is {adj_r2}")
        print(f"\tThe RMSE score is {rmse}")



        


