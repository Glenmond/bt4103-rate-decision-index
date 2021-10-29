import numpy as np
from datetime import datetime,timedelta, date
import datetime
# check your pickle compability, perhaps its pickle not pickle5
import pandas as pd
import pickle
import json

def isNaN(num):
    return num != num

#loading data
def load_market_data():
    market_data = pd.read_csv("web/data/ms_result.csv")
    return market_data

def load_ngram_market_data():
    in_year=2003
    file = open("web/data/st_df.pickle", "rb")
    st_df = pickle.load(file)
    out = st_df.loc[st_df.date.dt.year == in_year]
    file = open("web/data/mins_df.pickle", "rb")
    mins_df = pickle.load(file)
    out2 = mins_df.loc[mins_df.date.dt.year == in_year]
    return out, out2

def load_macro_data():
    gdp_sub_index = pd.read_csv("web/data/macro_gdp_data.csv")
    employment_sub_index = pd.read_csv("web/data/macro_employment_data.csv")
    inflation_sub_index = pd.read_csv("web/data/macro_inflation_data.csv")
    return gdp_sub_index, employment_sub_index, inflation_sub_index

def load_fff_data():
    fff_data = pd.read_csv("web/data/fff_result.csv")
    fake_data = pd.read_csv("web/data/macro_gdp_data.csv")
    
    return fff_data, fake_data

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

def clean_ngram_data(data1):
    all_words = ''
    for s in data1['lemmatizedSentences'].iteritems():
        words = ' '.join(s[1])
        all_words = all_words + words
        
    return all_words


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