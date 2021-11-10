import numpy as np
from datetime import datetime,timedelta, date
import datetime
# check your pickle compability, perhaps its pickle not pickle5
import pandas as pd
import pickle
import json

from web.macro_model_2.import_data import fetch_data
from pandas.tseries.offsets import MonthEnd
from fredapi import Fred

fred_api = "18fb1a5955cab2aae08b90a2ff0f6e42"
fred = Fred(api_key=fred_api)

def isNaN(num):
    return num != num

#loading data

#DONE updated data source
def load_market_data():
    final_pickle_directory = 'models/data/sentiment_data/historical/final_df.pickle'
    #minutes_pickle_directory = 'models/data/sentiment_data/historical/mins_df.pickle'
    #news_pickle_directory = 'models/data/sentiment_data/historical/news_df.pickle' 
    #file = open(statement_pickle_directory, "rb")
    #statement_df = pickle.load(file)
    #statement_df['date2'] = statement_df['date'].dt.strftime('%Y-%m')
    #file = open(minutes_pickle_directory, "rb")
    #mins_df = pickle.load(file)
    #file = open(news_pickle_directory, "rb")
    #news_df = pickle.load(file)
    return final_pickle_directory
    #return statement_pickle_directory, minutes_pickle_directory, news_pickle_directory

#DONE updated data source
def load_ngram_market_data():
    #in_year=year
    file = open('models/data/sentiment_data/historical/st_df.pickle', "rb")
    st_df = pickle.load(file)
    file = open('models/data/sentiment_data/historical/mins_df.pickle', "rb")
    mins_df = pickle.load(file)
    file = open("models/data/sentiment_data/historical/news_df.pickle", "rb")
    news_df = pickle.load(file)
    return st_df, mins_df, news_df

# for main dashboard gauge
def load_gauge_data():
    #in_year=year
    file = open('models/data/macroeconomic_indicators_data/macro_train_pred_pickle', "rb")
    train_df = pickle.load(file)
    
    file = open('models/data/macroeconomic_indicators_data/macro_test_pred_pickle', "rb")
    test_df = pickle.load(file)

    train_df.reset_index(inplace=True)
    train_df.rename(columns={'index': 'Date'}, inplace=True)
    train_df['Date'] = pd.to_datetime(train_df['Date']) + MonthEnd(0)

    test_df.reset_index(inplace=True)
    test_df.rename(columns={'index': 'Date'}, inplace=True)
    test_df['Date'] = pd.to_datetime(test_df['Date']) + MonthEnd(0)

    gauge_final_data = pd.concat([train_df, test_df])
    gauge_final_data['Date'] = gauge_final_data['Date'].dt.strftime('%Y-%m')
    gauge_final_data.reset_index(drop=True, inplace=True)
    
    fff_prob_data = pd.read_csv('models/data/fed_futures_data/latest/fff_raw_probs.csv', index_col=0)
    fff_prob_data['Date'] = pd.to_datetime(fff_prob_data['Date']) + MonthEnd(0)
    fff_prob_data['Date'] = fff_prob_data['Date'].dt.strftime('%Y-%m')
    
    return gauge_final_data, fff_prob_data

    
def load_macro_ts():
    df_trainres = pd.read_csv('web/data/overall_train_results.csv')
    df_testres = pd.read_csv('web/data/overall_test_results.csv')
    df_x_test = pd.read_csv('web/data/X_test_ME.csv') 
    df_x_train = pd.read_csv('web/data/X_train_ME.csv') 
    return df_trainres, df_testres, df_x_train, df_x_test
    
def load_macro_data():
    gdp_sub_index = pd.read_csv("models/data/macroeconomic_indicators_data/macro_gdp_data.csv")
    employment_sub_index = pd.read_csv("models/data/macroeconomic_indicators_data/macro_employment_data.csv")
    inflation_sub_index = pd.read_csv("models/data/macroeconomic_indicators_data/macro_inflation_data.csv")
    return gdp_sub_index, employment_sub_index, inflation_sub_index

def load_macro_model_data():
    try: # so that we do not need the FRED API call every time we try to access
        data = pd.read_pickle('web/macro_model_2/macro_data_pickle')
    except Exception:       
        data = fetch_data()
        data.to_pickle('web/macro_model_2/macro_data_pickle')
    return data

def load_fff_data(): 
    return 'models/data/fed_futures_data/latest/fff_result.csv'

def load_fff_vs_fomc_data():

    def clean_fomc_data(data):
        data = data.reset_index()
        data.columns = ['Date', 'Value']
        data['Date'] = pd.to_datetime(data['Date'])
        data = data.dropna()
        return data

    # low
    # Source: https://fred.stlouisfed.org/series/FEDTARRL
    low = fred.get_series('FEDTARRL')
    low = clean_fomc_data(low)

    # mid point
    # Source: https://fred.stlouisfed.org/series/FEDTARRM
    mid = fred.get_series('FEDTARRM')
    mid = clean_fomc_data(mid)

    # high
    # Source: https://fred.stlouisfed.org/series/FEDTARRH
    high = fred.get_series('FEDTARRH')
    high = clean_fomc_data(high)

    preds = pd.read_csv("models/data/fed_futures_data/latest/fff_preds.csv")
    return preds, [low, mid, high]

def load_dir_macro_values():
    y_train = 'models/data/macroeconomic_indicators_data/macro_y_train_pickle'
    y_test =  'models/data/macroeconomic_indicators_data/macro_y_test_pickle'
    x_train =  'models/data/macroeconomic_indicators_data/macro_X_train_pickle'
    x_test =  'models/data/macroeconomic_indicators_data/macro_X_test_pickle'
    
    return y_train, y_test, x_train, x_test

def load_dir_macro_ts():
    y_train = 'models/data/macroeconomic_indicators_data/macro_y_train_pickle'
    y_test =  'models/data/macroeconomic_indicators_data/macro_y_test_pickle'
    x_train =  'models/data/macroeconomic_indicators_data/macro_train_pred_pickle'
    x_test =   'models/data/macroeconomic_indicators_data/macro_test_pred_pickle'
    
    return y_train, y_test, x_train, x_test

#data preprocessing step (if required)
def clean_market(data):
    df = data
    #Statement
    list_statements = []
    for i in df.Score_Statement:
        if i > 0:
            list_statements.append("Hawkish")
        elif i < 0:
            list_statements.append("Dovish")
        elif i == 0:
            list_statements.append("Neutral")
        elif isNaN(i):
            list_statements.append("-")

    df['Statement_Sentiments'] = list_statements

    #Minutes
    list_minutes = []
    for j in df.Score_Minutes:
        if j > 0:
            list_minutes.append("Hawkish")
        elif j < 0:
            list_minutes.append("Dovish")
        elif j == 0:
            list_minutes.append("Neutral")
        elif isNaN(j):
            list_minutes.append("-")

    df['Minutes_Sentiments'] = list_minutes

    #News
    list_news = []
    for k in df.Score_News:
        if k > 0:
            list_news.append("Hawkish")
        elif k < 0:
            list_news.append("Dovish")
        elif k == 0:
            list_news.append("Neutral")
        elif isNaN(k):
            list_news.append("-")

    df['News_Sentiments'] = list_news
    
    return df

def clean_macro_ts(y_train, y_test):
    comb = [y_train, y_test]
    df_overall = pd.concat(comb)
    df_overall.reset_index(inplace=True)
    df_overall.rename(columns={"Unnamed: 0": "Date"}, inplace = True)
    return df_overall

def clean_maindashboard_macro(y_train, y_test, x_train, x_test):
    
    #import y val
    file_y_test = open(y_test, "rb") #"macro_y_test_pickle"
    y_test_df = pickle.load(file_y_test)
    file_y_train = open(y_train, "rb") #"macro_y_train_pickle"
    y_train_df = pickle.load(file_y_train)

    comb_y = [y_train_df, y_test_df]
    df_overall_y = pd.concat(comb_y)
    df_overall_y.reset_index(inplace=True)
    df_overall_y.rename(columns={"index": "Date"}, inplace = True)
    df_temp_y = df_overall_y.sort_values(by=['Date'])
    df_temp_y.reset_index(inplace=True)
    df_fin_y = df_temp_y[["Date", "target"]]
    
    #import feature (X) val
    file_X_test = open(x_test, "rb") #"macro_X_test_pickle"
    X_test_df = pickle.load(file_X_test)
    file_X_train = open(x_train, "rb") #"macro_X_train_pickle"
    X_train_df = pickle.load(file_X_train)

    comb_X = [X_train_df, X_test_df]
    df_overall_X = pd.concat(comb_X)
    df_overall_X.reset_index(inplace=True)
    df_overall_X.rename(columns={"index": "Date"}, inplace = True)
    df_temp_X = df_overall_X.sort_values(by=['Date'])
    df_temp_X.reset_index(inplace=True)
    df_fin_X = df_temp_X[["Date", "T10Y3M", "EMRATIO_MEDWAGES", "EMRATIO", "GDPC1", "MEDCPI", "MEDCPI_PPIACO", "HD_index", "shifted_target"]]
    
    #merging feature and predicted data
    df_plot = pd.merge(df_fin_X, df_fin_y, on="Date")
    
    ## ADD IN Date manipulation in necessary##
    return df_plot

def import_modify_pickle_overall_ts(y_train, y_test, pred_train, pred_test):
    
    #Predicted Value
    file_test_pred = open(pred_test, "rb") #"macro_test_pred_pickle"
    test_pred_df = pickle.load(file_test_pred)
    file_train_pred = open(pred_train, "rb") #"macro_train_pred_pickle"
    train_pred_df = pickle.load(file_train_pred)
    comb_pred = [train_pred_df, test_pred_df]
    df_pred = pd.concat(comb_pred)
    df_pred.reset_index(inplace=True)
    df_pred.rename(columns={"index": "Date", "Federal Funds Rate": "Predicted_Rate"}, inplace = True)
    df_temp_pred = df_pred.sort_values(by=['Date'])
    df_temp_pred.reset_index(inplace=True)
    df_fin_pred = df_temp_pred[["Date", "Predicted_Rate"]]
    df_fin_pred['Date'] = df_fin_pred['Date'].astype(str)
    
    #Actual Value
    file_y_test = open(y_test, "rb") #"macro_y_test_pickle"
    y_test_df = pickle.load(file_y_test)
    file_y_train = open(y_train, "rb") #"macro_y_train_pickle"
    y_train_df = pickle.load(file_y_train)
    comb_y = [y_train_df, y_test_df]
    df_overall_y = pd.concat(comb_y)
    df_overall_y.reset_index(inplace=True)
    df_overall_y.rename(columns={"index": "Date", "target": "Actual_Rate"}, inplace = True)
    df_temp_y = df_overall_y.sort_values(by=['Date'])
    df_temp_y.reset_index(inplace=True)
    df_fin_y = df_temp_y[["Date", "Actual_Rate"]]
    df_fin_y['Date'] = df_fin_y['Date'].astype(str)
    
    #Merge
    df_plot = pd.merge(df_fin_y, df_fin_pred, on="Date")
    
    return df_plot

#final
def import_modify_pickle_ms_main(file_final):
    
    ## Read from final pickle
    score_file = open(file_final, "rb") #"final_df.pickle"
    score_df = pickle.load(score_file)
    score_df.rename(columns={"date": "Date_Full"}, inplace = True)

    datel = []
    for d in score_df.Date_Full:
        datel.append(d.strftime("%Y-%m"))

    score_df['Date'] = datel

    dates = []
    for x in score_df.Date_Full:
        dates.append(x.strftime("%B %Y"))
    score_df['Date_Show'] = dates

    df = score_df[["Date", "Date_Full", "Date_Show", "Score_Statement", "Score_Minutes", "Score_News"]]
    
    #Add Sentiments
    #Statement
    list_statements = []
    for i in df.Score_Statement:
        if i > 0:
            list_statements.append("Hawkish")
        elif i < 0:
            list_statements.append("Dovish")
        elif i == 0:
            list_statements.append("Neutral")
        elif isNaN(i):
            list_statements.append("-")

    df['Statement_Sentiments'] = list_statements

    #Minutes
    list_minutes = []
    for j in df.Score_Minutes:
        if j > 0:
            list_minutes.append("Hawkish")
        elif j < 0:
            list_minutes.append("Dovish")
        elif j == 0:
            list_minutes.append("Neutral")
        elif isNaN(j):
            list_minutes.append("-")

    df['Minutes_Sentiments'] = list_minutes

    #News
    list_news = []
    for k in df.Score_News:
        if k > 0:
            list_news.append("Hawkish")
        elif k < 0:
            list_news.append("Dovish")
        elif k == 0:
            list_news.append("Neutral")
        elif isNaN(k):
            list_news.append("-")

    df['News_Sentiments'] = list_news
    
    return df


#helper function
def guess_date(string):
    for fmt in ["%Y-%m-%d"]:
        try:
            return datetime.datetime.strptime(string, fmt).date()
        except ValueError:
            continue
    raise ValueError(string)

def import_modify_csv_fff(file):
    df = pd.read_csv(file) 
    
    df.rename(columns={"Unnamed: 0": "Date"}, inplace = True)

    #Changing date format
    newdate = []
    for d in df.Date:
        newdate.append(guess_date(d).strftime("%Y-%m-%d"))
    newdate

    df['Date'] = newdate
    
    return df

def preprocessed_data(fff_data):
    """
    placeholder for cleaning data
    """
    cleaned_data = fff_data
    return cleaned_data