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
import plotly.io as pio
import json
from web.utils.utils import load_market_data, load_fff_data

#### gmond update this function
def plot_gauge(gauge_final_data, fff_prob_data, date):
    
    #do sth with date
    # macro + hd prediction rate
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = (gauge_final_data.iloc[(gauge_final_data.loc[gauge_final_data.Date == date].index).tolist()[0]].predicted).round(4),
        domain = {'row': 0, 'column': 0},
        title = {'text': "Rate Hike-Cut", 'font': {'size': 20}},
        delta = {'reference': (fff_prob_data.iloc[(fff_prob_data.loc[fff_prob_data.Date == date].index-1).tolist()[0]].predicted), 'increasing': {'color': "mediumseagreen"}},
        gauge = {
            'axis': {'range': [None, 1], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "white"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 0.1], 'color': 'maroon'},
                {'range': [0.1, 0.2], 'color': 'firebrick'},
                {'range': [0.2, 0.3], 'color': 'indianred'},
                {'range': [0.3, 0.4], 'color': 'lightcoral'},
                {'range': [0.4, 0.5], 'color': 'rosybrown'},
                {'range': [0.5, 0.6], 'color': 'floralwhite'},
                {'range': [0.6, 0.7], 'color': 'palegreen'},
                {'range': [0.7, 0.8], 'color': 'lightgreen'},
                {'range': [0.8, 0.9], 'color': 'limegreen'},
                {'range': [0.9, 1], 'color': 'forestgreen'}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 0.654}}))
    
    # fff probability
    fig.add_trace(go.Indicator(
        mode = "number",
        value = (fff_prob_data.iloc[(fff_prob_data.loc[fff_prob_data.Date == date].index).tolist()[0]].Hike),
        title = {'text':"Probability of Rate Change", 
                 'font.size': 20, 
                 'font.color': 'darkblue', 
                 'font.family':'Courier New'},
        domain = {'row': 1, 'column': 0}))

    #fig.update_layout(font = {'color': "darkblue", 'family': "Arial"})
    #fig.update_layout(width=800, height=450)
    plot_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    return plot_json


def plot_market(market_data, date):
    #currently is static and hard coded. 
    # need to implment date picker for entire dashboard for home page
    df_senti = market_data
    fig = go.Figure()
    #Statement
    fig.add_trace(go.Indicator(
        mode = "number+delta",
        value = (df_senti.iloc[(df_senti.loc[df_senti.Date == date].index).tolist()[0]].Score_Statement).round(4),
        delta = {'position': "right", 'reference': (df_senti.iloc[(df_senti.loc[df_senti.Date == date].index-1).tolist()[0]].Score_Statement).round(4)},
        title = {'text':"FOMC STATEMENT SENTIMENTS SCORE"+'<br>'+ '='*len("FOMC Statement Sentiments Score"), 
                 'font.size': 20, 
                 'font.color': 'darkblue', 
                 'font.family':'Courier New'},
        domain = {'row': 1, 'column': 0}))
    
    fig.add_trace(go.Indicator(
        title = {'text': "<"+df_senti.iloc[(df_senti.loc[df_senti.Date == date].index).tolist()[0]].Statement_Sentiments+">", 
                 'font.size': 17,
                 'font.family': 'Courier New', 
                 'align': 'right'},
        mode = 'delta',
        delta = {'reference': 0, 'font.size': 1},
        domain = {'row': 1, 'column': 1}))

    #Minutes
    fig.add_trace(go.Indicator(
        mode = "number+delta",
        value = (df_senti.iloc[(df_senti.loc[df_senti.Date == date].index).tolist()[0]].Score_Minutes).round(4),
        delta = {'position': "right", 'reference': (df_senti.iloc[(df_senti.loc[df_senti.Date == date].index-1).tolist()[0]].Score_Minutes).round(4)},
        title = {'text':"FOMC MINUTES SENTIMENTS SCORE"+'<br>'+ '='*len("FOMC Minutes Sentiments Score"),
                 'font.size': 20, 
                 'font.color': 'darkblue', 
                 'font.family':'Courier New'},
        domain = {'row': 3, 'column': 0}))
    
    fig.add_trace(go.Indicator(
        title = {'text': "<"+df_senti.iloc[(df_senti.loc[df_senti.Date == date].index).tolist()[0]].Minutes_Sentiments+">", 
                 'font.size': 17,
                 'font.family': 'Courier New', 
                 'align': 'right'},
        mode = 'delta',
        delta = {'reference': 0, 'font.size': 1},
        domain = {'row': 3, 'column': 1}))

    #News
    fig.add_trace(go.Indicator(
        mode = "number+delta",
        value = (df_senti.iloc[(df_senti.loc[df_senti.Date == date].index).tolist()[0]].Score_News).round(4),
        delta = {'position': "right", 'reference': (df_senti.iloc[(df_senti.loc[df_senti.Date == date].index-1).tolist()[0]].Score_News).round(4)},
        title = {'text':"NEWS SENTIMENTS SCORE"+'<br>'+ '='*len("News Sentiments Score"), 
                 'font.size': 20, 
                 'font.color': 'darkblue', 
                 'font.family':'Courier New'},
        domain = {'row': 5, 'column': 0}))
    
    fig.add_trace(go.Indicator(
        title = {'text': "<"+df_senti.iloc[(df_senti.loc[df_senti.Date == date].index).tolist()[0]].News_Sentiments+">", 
                 'font.size': 17,
                 'font.family': 'Courier New',
                 'align': 'right'},
        mode = 'delta',
        delta = {'reference': 0, 'font.size': 1},
        domain = {'row': 5, 'column': 1}))

    
    fig.update_layout(
        grid = {'rows': 6, 'columns': 2, 'pattern': "independent"},
        paper_bgcolor = "white", 
        font_family="Times New Roman Bold",
        font_color="black",
        height=600, width=800,
        title={
            'text': "Breakdown of Sentiment Scores for " + date,
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top', 
            'font.size':25},
        margin=dict(l=150, r=100, t=75, b=20))
    
    fig.update_layout(width=590, height=450)
    plot_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    return plot_json

def get_average_sentiment(market_data, date):
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
    
def plot_market_average(market_data, date):
    avg = get_average_sentiment(market_data, date).round(4)

    word = ""
    if avg > 0:
        word = "Overall Hawkish"
    elif avg < 0:
        word = "Overall Dovish"
    elif avg == 0:
        word = "Overall Neutral"
    
    fig = go.Figure()
    fig.add_trace(go.Indicator(
        title={
            'text': "Average Sentiment Scores for " + date,
            'font.size':30},
        mode = "delta",
        delta = {'reference': 0, 'font.size': 1},
        domain = {'row': 0, 'column': 0}))
    
    fig.add_trace(go.Indicator(
        mode = "number",
        value = avg,
        domain = {'row': 1, 'column': 0}))
    
    fig.add_trace(go.Indicator(
        title = {'text': "<"+word+">", 
                 'font.size': 30,
                 'font.family': 'Courier New Bold',
                 'font.color':'burlywood',            
                 'align': 'center'},
        mode = 'delta',
        delta = {'reference': 0, 'font.size': 1},
        domain = {'row': 2, 'column': 0}))
    
    
    fig.update_layout(
        grid = {'rows': 3, 'columns': 1, 'pattern': "independent"},
        paper_bgcolor = "steelblue", 
        font_family="Times New Roman Bold",
        font_color="black",
        height=600, width=800,
        margin=dict(l=5, r=5, t=25, b=5))

    
    fig.update_layout(width=600, height=300)
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
    
    fig.update_layout(width=600, height=550)
    plot_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return plot_json

def plot_macro_maindashboard(df_plot, date):
    #date= '2004-09-01'
    ## Setting up values
    #T10Y3M
    B_T10Y3M = 0.043143
    X_T10Y3M = df_plot.iloc[(df_plot.loc[df_plot.Date == date].index).tolist()[0]].T10Y3M #change to actual date to date
    value_T10Y3M = B_T10Y3M * X_T10Y3M

    X_T10Y3M_prev = df_plot.iloc[(df_plot.loc[df_plot.Date == date].index-1).tolist()[0]].T10Y3M
    value_T10Y3M_prev = B_T10Y3M * X_T10Y3M_prev
    
    #EMRATIO
    B_EMRATIO = 0.033783
    B_EMRATIO_MEDWAGES = 0.006322
    X_EMRATIO = df_plot.iloc[(df_plot.loc[df_plot.Date == date].index).tolist()[0]].EMRATIO #change to actual date to date
    X_EMRATIO_MEDWAGES = df_plot.iloc[(df_plot.loc[df_plot.Date == date].index).tolist()[0]].EMRATIO_MEDWAGES #change to actual date to date
    X_MEDWAGES = X_EMRATIO_MEDWAGES / X_EMRATIO
    value_EMRATIO = (B_EMRATIO + (B_EMRATIO_MEDWAGES*X_MEDWAGES)) * X_EMRATIO

    X_EMRATIO_prev = df_plot.iloc[(df_plot.loc[df_plot.Date == date].index-1).tolist()[0]].EMRATIO
    X_EMRATIO_MEDWAGES_prev = df_plot.iloc[(df_plot.loc[df_plot.Date == date].index-1).tolist()[0]].EMRATIO_MEDWAGES #change to actual date to date
    X_MEDWAGES_prev = X_EMRATIO_MEDWAGES_prev / X_EMRATIO_prev
    value_EMRATIO_prev = (B_EMRATIO + (B_EMRATIO_MEDWAGES*X_MEDWAGES_prev)) * X_EMRATIO_prev

    #GDP
    B_GDPC1 = 0.036187
    X_GDPC1 = df_plot.iloc[(df_plot.loc[df_plot.Date == date].index).tolist()[0]].GDPC1 #change to actual date to date
    value_GDPC1 = B_GDPC1 * X_GDPC1
    
    X_GDPC1_prev = df_plot.iloc[(df_plot.loc[df_plot.Date == date].index-1).tolist()[0]].GDPC1 #change to actual date to date
    value_GDPC1_prev = B_GDPC1 * X_GDPC1_prev
    
    #MEDCPI
    B_MEDCPI = 0.063183
    B_MEDCPI_PPIACO = -0.077871
    X_MEDCPI = df_plot.iloc[(df_plot.loc[df_plot.Date == date].index).tolist()[0]].MEDCPI #change to actual date to date
    X_MEDCPI_PPIACO = df_plot.iloc[(df_plot.loc[df_plot.Date == date].index).tolist()[0]].MEDCPI_PPIACO #change to actual date to date
    X_PPIACO = X_MEDCPI_PPIACO / X_MEDCPI
    value_MEDCPI = (B_MEDCPI + (B_MEDCPI_PPIACO*X_PPIACO)) * X_MEDCPI

    X_MEDCPI_prev = df_plot.iloc[(df_plot.loc[df_plot.Date == date].index-1).tolist()[0]].MEDCPI #change to actual date to date
    X_MEDCPI_PPIACO_prev = df_plot.iloc[(df_plot.loc[df_plot.Date == date].index-1).tolist()[0]].MEDCPI_PPIACO #change to actual date to date
    X_PPIACO_prev = X_MEDCPI_PPIACO_prev / X_MEDCPI_prev
    value_MEDCPI_prev = (B_MEDCPI + (B_MEDCPI_PPIACO*X_PPIACO_prev)) * X_MEDCPI_prev
    
    #HD index
    B_HD_index = 0.051086
    X_HD_index = df_plot.iloc[(df_plot.loc[df_plot.Date == date].index).tolist()[0]].HD_index #change to actual date to date
    value_HD_index = B_HD_index * X_HD_index
    
    X_HD_index_prev = df_plot.iloc[(df_plot.loc[df_plot.Date == date].index-1).tolist()[0]].HD_index #change to actual date to date
    value_HD_index_prev = B_HD_index * X_HD_index_prev
    
    #shifted_target
    B_shifted_target = 1.7117595779058272
    X_shifted_target = df_plot.iloc[(df_plot.loc[df_plot.Date == date].index).tolist()[0]].shifted_target #change to actual date to date
    value_shifted_target = B_shifted_target * X_shifted_target
    
    X_shifted_target_prev = df_plot.iloc[(df_plot.loc[df_plot.Date == date].index-1).tolist()[0]].shifted_target #change to actual date to date
    value_shifted_target_prev = B_shifted_target * X_shifted_target_prev

    
    ## PLotting figures
    fig = go.Figure()


    #T10Y3M Indicator
    fig.add_trace(go.Indicator(
        mode = "number+delta",
        value = abs(value_T10Y3M.round(4)),
        delta = {'position': "right", 'reference':abs(value_T10Y3M_prev.round(4))},
        title = {'text':"BOND YIELD SPREAD"+'<br>'+ '='*len("Bond Yield Spread"), 
                 'font.size': 20, 
                 'font.color': 'darkblue', 
                 'font.family':'Courier New'},
        domain = {'row': 0, 'column': 0}))

    #EMRATIO Indicator
    fig.add_trace(go.Indicator(
        mode = "number+delta",
        value = abs(value_EMRATIO.round(4)),
        delta = {'position': "right", 'reference':abs(value_EMRATIO_prev.round(4))},
        title = {'text':"EMPLOYMENT"+'<br>'+ '='*len("Employment"), 
                 'font.size': 20,
                 'font.color': 'darkblue', 
                 'font.family':'Courier New'},
        domain = {'row': 0, 'column': 1}))

    #GDP Indicator
    fig.add_trace(go.Indicator(
        mode = "number+delta",
        value = abs(value_GDPC1.round(4)),
        delta = {'position': "right", 'reference':abs(value_GDPC1_prev.round(4))},
        title = {'text':"DOMESTIC OUTPUT"+'<br>'+ '='*len("Domestic Output"), 
                 'font.size': 20,
                 'font.color': 'darkblue', 
                 'font.family':'Courier New'},
        domain = {'row': 1, 'column': 0}))

    #MEDCPI Indicator
    fig.add_trace(go.Indicator(
        mode = "number+delta",
        value = abs(value_MEDCPI.round(4)),
        delta = {'position': "right", 'reference':abs(value_MEDCPI_prev.round(4))},
        title = {'text':"INFLATION"+'<br>'+ '='*len("Inflation"), 
                 'font.size': 20,
                 'font.color': 'darkblue', 
                 'font.family':'Courier New'},
        domain = {'row': 1, 'column': 1}))

    #HD index Indicator
    fig.add_trace(go.Indicator(
        mode = "number+delta",
        value = abs(value_HD_index.round(4)),
        delta = {'position': "right", 'reference':abs(value_HD_index_prev.round(4))},
        title = {'text':"HAWKISH-DOVISH INDEX"+'<br>'+ '='*len("Hawisk-Dovish Index"), 
                 'font.size': 20,
                 'font.color': 'darkblue', 
                 'font.family':'Courier New'},
        domain = {'row': 2, 'column': 0}))

    #shifted target Indicator
    fig.add_trace(go.Indicator(
        mode = "number+delta",
        value = abs(value_shifted_target.round(4)),
        delta = {'position': "right", 'reference':abs(value_shifted_target_prev.round(4))},
        title = {'text':"PREVIOUS MONTH RATE"+'<br>'+ '='*len("Previous Month Rate"), 
                 'font.size': 20,
                 'font.color': 'darkblue', 
                 'font.family':'Courier New'},
        domain = {'row': 2, 'column': 1}))

    #Configure layout
    fig.update_layout(
        grid = {'rows': 3, 'columns': 2, 'pattern': "independent"},
        paper_bgcolor = "white", 
        font_family="Times New Roman Bold",
        font_color="black",
        width=590, height=470,
        title={
            'text': "Contribution of Indicators for " + date,
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top', 
            'font.size':25},
        margin=dict(l=50, r=50, t=100, b=15))
    
    fig.update_layout(width=590, height=450)
    plot_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return plot_json

def plot_contributions_pie(df_plot, date):
    ## Setting up values
    #T10Y3M
    B_T10Y3M = 0.043143
    X_T10Y3M = df_plot.iloc[(df_plot.loc[df_plot.Date == date].index).tolist()[0]].T10Y3M #change to actual date to date
    value_T10Y3M = B_T10Y3M * X_T10Y3M

    #EMRATIO
    B_EMRATIO = 0.033783
    B_EMRATIO_MEDWAGES = 0.006322
    X_EMRATIO = df_plot.iloc[(df_plot.loc[df_plot.Date == date].index).tolist()[0]].EMRATIO #change to actual date to date
    X_EMRATIO_MEDWAGES = df_plot.iloc[(df_plot.loc[df_plot.Date == date].index).tolist()[0]].EMRATIO_MEDWAGES #change to actual date to date
    X_MEDWAGES = X_EMRATIO_MEDWAGES / X_EMRATIO
    value_EMRATIO = (B_EMRATIO + (B_EMRATIO_MEDWAGES*X_MEDWAGES)) * X_EMRATIO

    #GDP
    B_GDPC1 = 0.036187
    X_GDPC1 = df_plot.iloc[(df_plot.loc[df_plot.Date == date].index).tolist()[0]].GDPC1 #change to actual date to date
    value_GDPC1 = B_GDPC1 * X_GDPC1

    #MEDCPI
    B_MEDCPI = 0.063183
    B_MEDCPI_PPIACO = -0.077871
    X_MEDCPI = df_plot.iloc[(df_plot.loc[df_plot.Date == date].index).tolist()[0]].MEDCPI #change to actual date to date
    X_MEDCPI_PPIACO = df_plot.iloc[(df_plot.loc[df_plot.Date == date].index).tolist()[0]].MEDCPI_PPIACO #change to actual date to date
    X_PPIACO = X_MEDCPI_PPIACO / X_MEDCPI
    value_MEDCPI = (B_MEDCPI + (B_MEDCPI_PPIACO*X_PPIACO)) * X_MEDCPI

    #HD index
    B_HD_index = 0.051086
    X_HD_index = df_plot.iloc[(df_plot.loc[df_plot.Date == date].index).tolist()[0]].HD_index #change to actual date to date
    value_HD_index = B_HD_index * X_HD_index

    #shifted_target
    B_shifted_target = 1.7117595779058272
    X_shifted_target = df_plot.iloc[(df_plot.loc[df_plot.Date == date].index).tolist()[0]].shifted_target #change to actual date to date
    value_shifted_target = B_shifted_target * X_shifted_target

    labels = ['Bond Yield Spread', 'Employment', 'Domestic Output', 'Inflation', 'Hawisk-Dovish Index', 'Previous Month Rate']
    values = [abs(value_T10Y3M), abs(value_EMRATIO), abs(value_GDPC1), abs(value_MEDCPI), abs(value_HD_index), abs(value_shifted_target)]
    colors = ['saddlebrown', 'salmon', 'peachpuff', 'palevioletred', 'rosybrown', 'mistyrose']

    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.5, 
                                 textfont = {'family':'Courier New', 'color':'white'},
                                 textfont_size=13,
                                 showlegend=False, 
                                 marker = dict(colors=colors,line=dict(color='#000000', width=0.2), ),
                                 insidetextorientation='radial'
                                 )])
    fig.update_traces(textposition='outside', textinfo='label+percent',insidetextorientation='radial')
    fig.update_layout(
        paper_bgcolor = "steelblue",
        font_color="black",
        title={
            'text': "Weightage of Contributions of Respective Indicators on " + date,
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top', 
            'font.size':18,
            'font.family': 'Times New Roman Bold'},
        margin=dict(l=10, r=10, t=50, b=10),)

    
    fig.update_layout(width=600, height=300)
    plot_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return plot_json

def plot_fed_rates_ts(data):
    df_plot=data
    plot = go.Figure(data=[go.Scatter(
        name='Actual',
        x=df_plot.Date.tolist(),
        y=df_plot.actual_values.tolist(),
        marker_color='#FA8072' #change color of line
    ),
        go.Scatter(
        name='Predicted',
        x=df_plot.Date.tolist(),
        y=df_plot.predicted.tolist(),
        marker_color='#4682B4' #change color of line
    )
    ])

    plot.update_layout(
        updatemenus=[
            dict(
                active=0,
                buttons=list([
                    dict(label="Both",
                         method="update",
                         args=[{"visible": [True, True]},
                               {"title": "Time Series of Both Actual and Predicted Federal Funds Rates"}]),
                    dict(label="Actual",
                         method="update",
                         args=[{"visible": [True, False]},
                               {"title": "Time Series of Actual Federal Funds Rates",
                                }]),
                    dict(label="Predicted",
                         method="update",
                         args=[{"visible": [False, True]},
                               {"title": "Time Series of Predicted Federal Funds Rates",
                                }]),
                ]),
            )
        ])

    plot.update_layout(
            font_family="Courier New",
            font_color="black",
            title_font_family="Times New Roman Bold",
            title_font_color="black",
            title_text='Time Series of Both Actual and Predicted Federal Funds Rates', 
            title_x=0.5
        )
    plot.update_xaxes(rangeslider_visible=True)

    plot_json = json.dumps(plot, cls=plotly.utils.PlotlyJSONEncoder)
    return plot_json

