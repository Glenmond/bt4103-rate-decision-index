import numpy as np
from datetime import datetime,timedelta, date
import datetime
# check your pickle compability, perhaps its pickle not pickle5
import pandas as pd
import pickle
import json
from web.macro_model_2.import_data import fetch_data

from fredapi import Fred
fred_api = "18fb1a5955cab2aae08b90a2ff0f6e42"
fred = Fred(api_key=fred_api)

def isNaN(num):
    return num != num

#loading data
def load_market_data():
    #market_data = pd.read_csv("web/data/ms_result.csv")
    statement_pickle_directory = '../analytics/data/sentiment_data/historical/st_df.pickle'

    minutes_pickle_directory = '../analytics/data/sentiment_data/historical/mins_df.pickle'

    news_pickle_directory = '../analytics/data/sentiment_data/historical/news_df.pickle' 
    file = open(statement_pickle_directory, "rb")
    statement_df = pickle.load(file)
    file = open(minutes_pickle_directory, "rb")
    mins_df = pickle.load(file)
    file = open(news_pickle_directory, "rb")
    news_df = pickle.load(file)
    
    return statement_pickle_directory, minutes_pickle_directory, news_pickle_directory

def load_ngram_market_data(year):
    in_year=year
    file = open("web/data/st_df.pickle", "rb")
    mins_df = pickle.load(file)
    out = mins_df.loc[mins_df.date.dt.year == in_year]
    file = open("web/data/mins_df.pickle", "rb")
    mins_df = pickle.load(file)
    out2 = mins_df.loc[mins_df.date.dt.year == in_year]
    #file = open("web/data/news_df.pickle", "rb")
    #mins_df = pickle.load(file)
    #out3 = mins_df.loc[mins_df.date.dt.year == in_year]
    #return out, out2, out3
    return out, out2
    
def load_macro_ts():
    df_trainres = pd.read_csv('web/data/overall_train_results.csv')
    df_testres = pd.read_csv('web/data/overall_test_results.csv')
    df_x_test = pd.read_csv('web/data/X_test_ME.csv') 
    df_x_train = pd.read_csv('web/data/X_train_ME.csv') 
    return df_trainres, df_testres, df_x_train, df_x_test
    
def load_macro_data():
    gdp_sub_index = pd.read_csv("web/data/macro_gdp_data.csv")
    employment_sub_index = pd.read_csv("web/data/macro_employment_data.csv")
    inflation_sub_index = pd.read_csv("web/data/macro_inflation_data.csv")
    return gdp_sub_index, employment_sub_index, inflation_sub_index

def load_macro_model_data():
    try: # so that we do not need the FRED API call every time we try to access
        data = pd.read_pickle('web/macro_model_2/macro_data_pickle')
    except Exception:       
        data = fetch_data()
        data.to_pickle('web/macro_model_2/macro_data_pickle')
    return data

def load_fff_data():
    fff_data = pd.read_csv("web/data/fff_result.csv")    
    return fff_data

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

    preds = pd.read_csv("web/data/fff_preds.csv")
    return preds, [low, mid, high]

def load_home_data():
    home_data = pd.read_csv("web/data/macro_gdp_data.csv")
    return home_data


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
    #import predicted data
    #df_trainres_pred = pd.read_csv(y_train) #y_train  ('overall_train_results.csv')
    #df_testres_pred = pd.read_csv(y_test) #y_test  ('overall_test_results.csv')
    comb_pred = [y_train, y_test]
    df_overall_pred = pd.concat(comb_pred)
    df_overall_pred.reset_index(inplace=True)
    df_overall_pred.rename(columns={"Unnamed: 0": "Date"}, inplace = True)

    df_overall_pred['Date'] = pd.to_datetime(df_overall_pred['Date'])
    df_temp_pred = df_overall_pred.sort_values(by=['Date'])
    df_temp_pred.reset_index(inplace=True)
    df_fin_pred = df_temp_pred[["Date", "actual_values", "predicted"]]
    
    #import feature data
    #df_trainres = pd.read_csv(x_train) #x_train  ('X_train_ME.csv')
    #df_testres = pd.read_csv(x_test) #x_test  ('X_test_ME.csv')
    comb = [x_train, x_test]
    df_overall = pd.concat(comb)
    df_overall.reset_index(inplace=True)
    df_overall.rename(columns={"Unnamed: 0": "Date"}, inplace = True)
    df_overall['Date'] = pd.to_datetime(df_overall['Date'])
    df_temp = df_overall.sort_values(by=['Date'])
    df_temp2 = df_temp[["Date", "T10Y3M", "EMRATIO_MEDWAGES", "EMRATIO", "GDPC1", "MEDCPI", "MEDCPI_PPIACO", "HD_index", "shifted_target"]]
    df_temp2.reset_index(inplace=True)
    df_fin = df_temp2[["Date", "T10Y3M", "EMRATIO_MEDWAGES", "EMRATIO", "GDPC1", "MEDCPI", "MEDCPI_PPIACO", "HD_index", "shifted_target"]]
    # df_fin2 = df_fin.set_index('Date')
    
    #merging feature and predicted data
    df_plot = pd.merge(df_fin, df_fin_pred, on="Date")
    # df_plot_fin = df_plot.set_index('Date')
    df_plot['Date'] = df_plot['Date'] - pd.tseries.offsets.MonthEnd(-1)
    return df_plot

def import_modify_pickle_ms_main(file_st, file_mins, file_news):
    #Statement
    st_file = open(file_st, "rb") #"st_df.pickle"
    st_df = pickle.load(st_file)
    st_df_to_comb = st_df[["date", "Scaled Score"]]
    st_df_to_comb['date'] = st_df_to_comb['date'] - pd.tseries.offsets.MonthEnd(-1)    
    st_df_to_comb.rename(columns={"date": "Date", "Scaled Score": "Score_Statement"}, inplace = True)
    
    #Minutes
    mins_file = open(file_mins, "rb") #"mins_df.pickle"
    mins_df = pickle.load(mins_file)
    mins_df_to_comb = mins_df[["date", "Scaled Score"]]
    mins_df_to_comb['date'] = mins_df_to_comb['date'] - pd.tseries.offsets.MonthEnd(-1)    
    mins_df_to_comb.rename(columns={"date": "Date", "Scaled Score": "Score_Minutes"}, inplace = True)

    #News
    news_file = open(file_news, "rb") #"news_df.pickle"
    news_df = pickle.load(news_file)
    news_df_to_comb = news_df[["date", "Scaled Score"]] 
    news_df_to_comb.rename(columns={"date": "Date", "Scaled Score": "Score_News"}, inplace = True)
    news_df_to_comb = news_df_to_comb[1:]
    
    #Join Statement and Minutes
    df_temp = pd.merge(st_df_to_comb, mins_df_to_comb, on="Date", how='inner')
    
    
    #Join df_temp and News
    df = pd.merge(df_temp, news_df_to_comb, on="Date", how='inner')    
    
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
    for fmt in ["%d/%m/%y"]:
        try:
            return datetime.datetime.strptime(string, fmt).date()
        except ValueError:
            continue
    raise ValueError(string)

def clean_fff(fff_data):
    #Changing date format
    df = fff_data
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