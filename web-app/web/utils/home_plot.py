import pandas as pd
import numpy as np
import dateutil
import datetime
import numpy as np
import pandas as pd
import plotly
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from vega_datasets import data
from plotly.subplots import make_subplots
import plotly.express as px
import json
from web.utils.utils import load_home_data, load_market_data, load_fff_data

def plot_market(market_data):
    #currently is static and hard coded. 
    # need to implment date picker for entire dashboard for home page
    date = '2000-04-30'
    df_senti = market_data
    fig = go.Figure()
    
    fig.add_trace(go.Indicator(
        title = {'text': "Sentiments for " + date, 
                 'font.size': 30,
                 'align': 'center'},
        mode = 'delta',
        delta = {'reference': 0, 'font.size': 1},
        domain = {'row': 0, 'column': 0}))
    
    #Statement
    fig.add_trace(go.Indicator(
        mode = "number+delta",
        value = df_senti.iloc[(df_senti.loc[df_senti.Date == date].index).tolist()[0]].Score_Statement,
        delta = {'position': "bottom", 'reference': df_senti.iloc[(df_senti.loc[df_senti.Date == date].index-1).tolist()[0]].Score_Statement},
        title = {'text':"FOMC Statement Sentiments Score", 
                 'font.size': 20},
        domain = {'row': 1, 'column': 0}))
    
    fig.add_trace(go.Indicator(
        title = {'text': df_senti.iloc[(df_senti.loc[df_senti.Date == date].index).tolist()[0]].Statement_Sentiments, 
                 'font.size': 15,
                 'font.family': 'Courier New', 
                 'align': 'left'},
        mode = 'delta',
        delta = {'reference': 0, 'font.size': 1},
        domain = {'row': 1, 'column': 1}))

    #Minutes
    fig.add_trace(go.Indicator(
        mode = "number+delta",
        value = df_senti.iloc[(df_senti.loc[df_senti.Date == date].index).tolist()[0]].Score_Minutes,
        delta = {'position': "bottom", 'reference': df_senti.iloc[(df_senti.loc[df_senti.Date == date].index-1).tolist()[0]].Score_Minutes},
        title = {'text':"FOMC Minutes Sentiments Score",
                 'font.size': 20},
        domain = {'row': 2, 'column': 0}))
    
    fig.add_trace(go.Indicator(
        title = {'text': df_senti.iloc[(df_senti.loc[df_senti.Date == date].index).tolist()[0]].Minutes_Sentiments, 
                 'font.size': 15,
                 'font.family': 'Courier New', 
                 'align': 'left'},
        mode = 'delta',
        delta = {'reference': 0, 'font.size': 1},
        domain = {'row': 2, 'column': 1}))

    #News
    fig.add_trace(go.Indicator(
        mode = "number+delta",
        value = df_senti.iloc[(df_senti.loc[df_senti.Date == date].index).tolist()[0]].Score_News,
        delta = {'position': "bottom", 'reference': df_senti.iloc[(df_senti.loc[df_senti.Date == date].index-1).tolist()[0]].Score_News},
        title = {'text':"News Sentiments Score", 
                 'font.size': 20},
        domain = {'row': 3, 'column': 0}))
    
    fig.add_trace(go.Indicator(
        title = {'text': df_senti.iloc[(df_senti.loc[df_senti.Date == date].index).tolist()[0]].News_Sentiments, 
                 'font.size': 15,
                 'font.family': 'Courier New',
                 'align': 'left'},
        mode = 'delta',
        delta = {'reference': 0, 'font.size': 1},
        domain = {'row': 3, 'column': 1}))

    
    
    fig.update_layout(
        grid = {'rows': 4, 'columns': 2, 'pattern': "independent"}, 
        paper_bgcolor = "white", 
        font_family="Times New Roman Bold",
        font_color="black")
    
    fig.update_layout(width=800, height=450)
    plot_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    return plot_json

def get_average_sentiment(market_data):
    date = '2000-04-30'
    df_senti = market_data
    #    res = 0
    state_num = df_senti.iloc[(df_senti.loc[df_senti.Date == date].index).tolist()[0]].Score_Statement
    min_num = df_senti.iloc[(df_senti.loc[df_senti.Date == date].index).tolist()[0]].Score_Minutes
    news_num = df_senti.iloc[(df_senti.loc[df_senti.Date == date].index).tolist()[0]].Score_News
    
#     if isNaN(state_num):
#         res = ((min_num + news_num) / 2)
        
#     elif isNaN(min_num):
#         res = ((state_num + news_num) / 2)
        
#     elif isNaN(news_num):
#         res = ((state_num + min_num) / 2)
        
#     elif isNaN(state_num) and isNaN(min_num):
#         res = news_num
        
#     elif isNaN(state_num) and isNaN(news_num):
#         res = min_num
        
#     elif isNaN(min_num) and isNaN(news_num):
#         res = state_num
        
#     elif isNaN(min_num) and isNaN(news_num) and isNaN(state_num):
#         res = 0
    
#     else:
#         res = ((state_num+min_num+news_num) / 3)
    
    return ((state_num+min_num+news_num) / 3)
    

def plot_market_average(market_data):
    date = '2000-04-30'
    df_senti = market_data
    
    avg = get_average_sentiment(market_data)
    
    word = ""
    if avg > 0:
        word = "Overall Hawkish"
    elif avg < 0:
        word = "Overall Dovish"
    elif avg == 0:
        word = "Overall Neutral"
    
    fig = go.Figure()
    fig.add_trace(go.Indicator(
        mode = "number",
        value = avg,
        number = {'valueformat':'a'},
        title = {'text': "Average Sentiment Score for" + " " + date, 
                 'font.size': 25,
                 'font.family': 'Times New Roman Bold'},
        domain = {'row': 0, 'column': 0}))
    
    fig.add_trace(go.Indicator(
        title = {'text': word, 
                 'font.size': 20,
                 'font.family': 'Courier New'},
        mode = 'delta',
        delta = {'reference': 0, 'font.size': 1},
        domain = {'row': 0, 'column': 1}))
    
    fig.update_layout(
        grid = {'rows': 1, 'columns': 2, 'pattern': "independent"}, 
        paper_bgcolor = "white", 
        font_family="Times New Roman Bold",
        font_color="black")
    
    
    fig.update_layout(width=650, height=200)
    plot_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return plot_json
       
def plot_fff(fff_data):
    
    date = '2022-09-20'
    df_fff =fff_data
    #transform df
    new_df_fff = df_fff.melt(id_vars=["Date"],
                                   var_name="Basis Points",
                                   value_name="Probability")
    fig = px.bar(new_df_fff, x='Basis Points', y='Probability', animation_frame='Date')

    fig.update_layout(
        font_family="Courier New",
        font_color="black",
        title_font_family="Times New Roman Bold",
        title_font_color="black",
        title_text='Probability of Change in Target Rates by Basis Points', 
        title_x=0.5, 
        title_font_size = 20,
        xaxis_title="Amount of Change in Target Rates",
        yaxis_title="Probability of Change"
    )

    fig.update_yaxes(range=[-1.1, 1.1])
    
    #fig.update_layout(width=650, height=200)
    plot_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return plot_json

def plot_gdp_index(home_data):
    '''
    Placeholder function for plots
    '''
    df = home_data
    x = df['Date']

    fig = go.Figure()

    fig.add_trace(go.Line(x=x, y=df['GDPC1'],name='GDP', marker=dict(color="Black"))),
    fig.add_trace(go.Line(x=x, y=df['PCEC96'],name='Domestic Consumption', marker=dict(color="lightcoral"))),
    fig.add_trace(go.Line(x=x, y=df['GPDIC1'], name='Domestic Investment', marker=dict(color="sandybrown")))
    fig.add_trace(go.Line(x=x, y=df['GCEC1'], name='Government Expenditure', marker=dict(color="darkseagreen")))
    fig.add_trace(go.Line(x=x, y=df['NETEXC'], name='Net Export', marker=dict(color="cornflowerblue")))

    fig.update_layout(title_text='GDP and its components',
                    xaxis_title='Date', yaxis_title='Value')
    fig.update_layout(
        updatemenus=[
            dict(active=0,
                buttons=list([
                dict(label="All",
                    method="update",
                    args=[{"visible":[True,True,True,True,True]},
                        {"title":"ALL"}]),
                dict(label="Domestic Consumption",
                    method="update",
                    args=[{"visible":[True, True, False,False,False]},
                        {"title":"Domestic Consumption"}]),
                dict(label="Domestic Investment",
                    method="update",
                    args=[{"visible":[True,False,True,False,False]},
                        {"title":"Domestic Investment"}]),
                dict(label="Government Expenditure",
                    method="update",
                    args=[{"visible":[True,False,False,True,False]},
                        {"title":"Government Expenditure"}]),
                dict(label="Net Export",
                    method="update",
                    args=[{"visible":[True,False,False,False,True]},
                        {"title":"Net Export"}])
            ]),
            )
        ],
    )
    #fig.update_layout(width=600, height=400)
    plot_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return plot_json