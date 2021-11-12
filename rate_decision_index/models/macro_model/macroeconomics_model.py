import pandas as pd
import numpy as np
import math
import datetime
from dateutil.relativedelta import relativedelta
from scipy.stats.mstats import hmean

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
import statsmodels.api as sm

from fredapi import Fred
fred_api = "18fb1a5955cab2aae08b90a2ff0f6e42"
fred = Fred(api_key=fred_api)

import warnings 
warnings.filterwarnings('ignore')

from .base_model import Model

class MacroData():
    def __init__(self, data, path_to_HD_pickle = './data/sentiment_data/historical/'):
        self.data = data

        # Initialise train and test sets as None first
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None

        # Initialise the scaler as None first
        self.scaler=None
        self.imputer=None

        self.preprocess(path_to_HD_pickle)

    def preprocess(self, path_to_HD_pickle):
        """
        Preprocess the data obtained from the API call
        """
        # Add the HD index
        hd = pd.read_pickle(path_to_HD_pickle + 'final_df.pickle')
        hd.index = hd['date']
        hd.index = pd.to_datetime(hd.index).to_period('M')
        hd = hd.drop("date", axis=1)
        # hd = hd.shift(1) # Last month's sentiments affect this month's rate
        hd = hd[hd.index >= '2003-01']
        #hd = hd[hd.index <= '2021-04']

        #hd_imputer = KNNImputer(n_neighbors=7)
        #hd = pd.DataFrame(hd_imputer.fit_transform(hd), index=hd.index, columns=hd.columns)
        hd = hd.fillna(method="ffill")

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
        #self.data.dropna(inplace=True)

        # Add interactions
        self.data = self.data.fillna(method="ffill")
        self.data = self.data.fillna(method="bfill")

        # Do train test split
        X = self.data.copy().drop('target', axis=1)
        y = pd.DataFrame(self.data['target'])
        test_proportion = int(0.1*len(self.data))
        X_train = X[:-test_proportion]
        y_train = y[:-test_proportion]
        self.X_test = X[-test_proportion:]
        self.y_test = y[-test_proportion:]
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=42)        

        # Add interactions
        self.X_train['MEDCPI_PPIACO'] = self.X_train['MEDCPI'] * self.X_train['PPIACO']
        self.X_train['EMRATIO_MEDWAGES'] = self.X_train['EMRATIO'] * self.X_train['MEDWAGES']
        self.X_val['MEDCPI_PPIACO'] = self.X_val['MEDCPI'] * self.X_val['PPIACO']
        self.X_val['EMRATIO_MEDWAGES'] = self.X_val['EMRATIO'] * self.X_val['MEDWAGES']
        self.X_test['MEDCPI_PPIACO'] = self.X_test['MEDCPI'] * self.X_test['PPIACO']
        self.X_test['EMRATIO_MEDWAGES'] = self.X_test['EMRATIO'] * self.X_test['MEDWAGES']

        # Do some transformations
        self.X_train['HD_index'] = np.power(self.X_train['HD_index'],1/2)
        self.X_test['HD_index'] = np.power(self.X_test['HD_index'],1/2)
        self.X_val['HD_index'] = np.power(self.X_val['HD_index'],1/2)

        # Rearrange the input features
        self.X_train = self.X_train[['T10Y3M', 'EMRATIO_MEDWAGES','EMRATIO', 'GDPC1','MEDCPI','MEDCPI_PPIACO','HD_index','shifted_target']]
        self.X_val = self.X_val[['T10Y3M', 'EMRATIO_MEDWAGES','EMRATIO', 'GDPC1','MEDCPI','MEDCPI_PPIACO','HD_index','shifted_target']]
        self.X_test = self.X_test[['T10Y3M', 'EMRATIO_MEDWAGES','EMRATIO', 'GDPC1','MEDCPI','MEDCPI_PPIACO','HD_index','shifted_target']]

        # Do some scalings
        scaler = StandardScaler()
        scaler.fit(self.X_train) # fit the scaler only on training set
        self.X_train = pd.DataFrame(scaler.transform(self.X_train), columns=self.X_train.columns, index=self.X_train.index)
        self.X_val = pd.DataFrame(scaler.transform(self.X_val), columns=self.X_val.columns, index=self.X_val.index)
        self.X_test = pd.DataFrame(scaler.transform(self.X_test), columns=self.X_test.columns, index=self.X_test.index)
        self.scaler = scaler

class MacroModel(Model):
    def __init__(self, data: MacroData, shift_coef = 1.7119430641311288):
        self.data = data
        self.scaler = data.scaler
        self.imputer = data.imputer
        self.shift_coef = shift_coef

        # Initialize model results as None first
        self.fitted_model = None
        self.model_details = None

        # Initialize test and val predictions as None first
        self.validation_results = None
        self.testing_results = None      

    def fit_data(self):
        """
        Fits the data onto the model
        """
        exog = self.data.X_train.copy().drop('shifted_target', axis=1)
        endog = self.data.y_train

        try:
            # add a constant term for the regression
            exog = sm.add_constant(exog) 
            # this is the same as OLS
            model = sm.GLM(endog,exog, family = sm.families.Gaussian(sm.families.links.identity()), 
                            offset = self.shift_coef * self.data.X_train['shifted_target'] )
            model_results = model.fit()
            self.fitted_model = model_results
            self.model_details = model_results.summary()
        except (TypeError):
            print("The dataset is not defined properly, make sure you preprocess the data first before fitting!") 

    def predict(self, data):
        """
        Predicts the data using the fitted model
        """
        try:
            data_to_pred = data.copy().drop('shifted_target', axis=1)

            y_pred_res = self.fitted_model.get_prediction(sm.add_constant(data_to_pred), offset = self.shift_coef * data['shifted_target'])
            y_pred = y_pred_res.predicted_mean
            y_pred = np.array([[x] for x in y_pred])
            y_pred = [np.array([0]) if x < 0 else x for x in y_pred]
            
            return y_pred_res, y_pred
        except Exception as e:
            print(e)
            print("The dataset is not defined properly, make sure you fit the model with data first before fitting!") 
            return

    def assess_val_set_performance(self):
        """
        Gets prediction result on validation set
        """
        try:
            y_pred_res, y_pred = self.predict(self.data.X_val)
            self.validation_results = y_pred_res
        except Exception as e:
            print("Error when obtaining prediction for validation set") 
            print(e)
            print()
            return

        r2 = r2_score(self.data.y_val, y_pred)
        adj_r2 = 1-(1-r2)*(len(self.data.X_val)-1)/(len(self.data.X_val)-4-1)
        rmse = math.sqrt(mean_squared_error(self.data.y_val, y_pred))

        print("Performance on Validation Set")
        print(f"\tThe R2 score is {r2}")
        print(f"\tThe Adjusted R2 score is {adj_r2}")
        print(f"\tThe RMSE score is {rmse}")
        
    def assess_test_set_performance(self):
        """
        Gets prediction result on test set
        """
        try:
            y_pred_res, y_pred = self.predict(self.data.X_test)
            self.testing_result = y_pred_res
        except Exception as e:
            print("Error when obtaining prediction for test set") 
            print(e)
            print()
            return
        
        r2 = r2_score(self.data.y_test, y_pred)
        adj_r2 = 1-(1-r2)*(len(self.data.X_test)-1)/(len(self.data.X_test)-4-1)
        rmse = math.sqrt(mean_squared_error(self.data.y_test, y_pred))

        print("Performance on Test Set")
        print(f"\tThe R2 score is {r2}")
        print(f"\tThe Adjusted R2 score is {adj_r2}")
        print(f"\tThe RMSE score is {rmse}")

    def predict_latest_data(self, path_to_HD_pickle ='./data/sentiment_data/historical/'):
        """
        Gets the latest available data from FRED and returns the predicted data based on it. If the latest data for this particular month is not available,
        the next latest is obtained. For example, if Nov 2021 data is not available, the next latest available (maybe Sept 2021 or something) will be used
        For indicators with lead/lag periods, we will adjust accordingly if possible. 
        """
        most_recent_date = datetime.datetime.today()
        # Concurrent
        commodities = ('PPIACO', {'observation_end':most_recent_date.strftime('%Y-%m-%d'),
                                    'units':'lin',
                                    'frequency':'m',
                                    'aggregation_method': 'eop',   
                                    })

        # For indicators with lag period, there is no way of getting future data to shift backwards, so we just make do with present data                          
        real_gdp = ('GDPC1', {'observation_end':most_recent_date.strftime('%Y-%m-%d'),
                                'units':'lin',
                                'frequency':'q',
                                'aggregation_method': 'eop'
                                })
        median_cpi = ('MEDCPIM158SFRBCLE', {'observation_end':most_recent_date.strftime('%Y-%m-%d'),
                                            'units':'lin',
                                            'frequency':'m',
                                            'aggregation_method': 'eop',   
                                            })
        em_ratio = ('EMRATIO', {'observation_end':most_recent_date.strftime('%Y-%m-%d'),
                                'units':'lin',
                                'frequency':'m',
                                'aggregation_method': 'eop',   
                                })
        med_wages = ('LES1252881600Q', {'observation_end': most_recent_date.strftime('%Y-%m-%d'),
                                        'units':'lin',
                                        'frequency':'q',
                                        'aggregation_method': 'eop',   
                                        })

        # For leading indicators, we align by getting the most recent data n months ago, with n being the amount of lead associated with the indicator
        maturity_minus_three_month = ('T10Y3M', {'observation_end':(most_recent_date - relativedelta(months=5)).strftime('%Y-%m-%d'),
                                                'units':'lin',
                                                'frequency':'m',
                                                'aggregation_method': 'eop',
                                                })
        
        indicators = [commodities,real_gdp,median_cpi, em_ratio,
                    med_wages, maturity_minus_three_month]

        df = pd.DataFrame()

        for series_id, params in indicators:
            # Get the data from FRED, convert to pandas DataFrame
            indicator = fred.get_series(series_id, **params)
            indicator = indicator.to_frame().set_axis([series_id], axis="columns")
            # fill in data with '0.0' that is presented as just '.'
            indicator[series_id] = ["0.0" if x == "." else x for x in indicator[series_id]]
            # turn the value into numeric
            indicator[series_id] = pd.to_numeric(indicator[series_id])
            indicator.index = pd.to_datetime(indicator.index).to_period("M")
            indicator = indicator.resample("M").interpolate()

            if series_id in ("MEDCPIM158SFRBCLE"):  
                indicator.rename(columns={"MEDCPIM158SFRBCLE": "MEDCPI"}, inplace=True)

            if series_id in ("LES1252881600Q"):  # align 5 lag
                indicator = indicator.shift(-5)[:-5]
                indicator.rename(columns={"LES1252881600Q": "MEDWAGES"}, inplace=True)
            indicator= indicator.dropna()[-12:] # get 1 year of data first
            indicator = indicator.reset_index()
            indicator_name = indicator.columns[-1]
            df[indicator_name] = indicator[indicator_name]
        
        # add shifted indicator
        fed_fund_rate = fred.get_series(
            "DFF",
            **{"observation_end": (most_recent_date-relativedelta(months=2)).strftime('%Y-%m-%d'), # get the latest available end-of-month data, then shift by 1, so total 2 shifts
                "frequency": "m",
                "aggregation_method": "eop",
            }
        )
        fed_fund_rate.index = pd.to_datetime(fed_fund_rate.index).to_period("M")
        fed_fund_rate = fed_fund_rate[-13:]
        fed_fund_rate = fed_fund_rate.shift(1).dropna()
        df["shifted_target"] = fed_fund_rate.to_numpy()        

        # Add interactions
        df['MEDCPI_PPIACO'] = df['MEDCPI'] * df['PPIACO']
        df['EMRATIO_MEDWAGES'] = df['EMRATIO'] * df['MEDWAGES']

        # Add the HD index
        hd = pd.read_pickle(path_to_HD_pickle + 'final_df.pickle')
        hd.index = hd['date']
        hd.index = pd.to_datetime(hd.index).to_period('M')
        hd = hd.drop("date", axis=1)
        hd['Score_Statement'] = MinMaxScaler().fit_transform(hd['Score_Statement'].values.reshape(-1,1))
        hd['Score_Minutes'] = MinMaxScaler().fit_transform(hd['Score_Minutes'].values.reshape(-1,1))
        hd['Score_News'] = MinMaxScaler().fit_transform(hd['Score_News'].values.reshape(-1,1))
        hd['Overall'] = hmean([hd['Score_Statement'],hd['Score_Minutes'],hd['Score_News']])
        
        df['HD_index'] = hd[-12:].reset_index()['Overall']
        df['HD_index'] = np.power(df['HD_index'],1/2)       

        df = df[['T10Y3M', 'EMRATIO_MEDWAGES','EMRATIO', 'GDPC1','MEDCPI','MEDCPI_PPIACO','HD_index','shifted_target']]

        df = pd.DataFrame(self.scaler.transform(df), columns=df.columns, index=df.index)

        latest_prediction_results, latest_prediction_value = self.predict(df)
        return latest_prediction_value[-1][0]
