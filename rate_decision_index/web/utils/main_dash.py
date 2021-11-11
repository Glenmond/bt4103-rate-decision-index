# this is where we do our main analysis using our models and data
import numpy as np
import pickle
import math
from sklearn.metrics import r2_score, mean_squared_error

import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output

import pandas as pd
import plotly.graph_objs as go

from plotly.subplots import make_subplots
import plotly.graph_objects as go

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
import os


# IMPORT ALL THE DATASET
from altair import Chart, X, Y, Axis, Data, DataFormat
import pandas as pd
import numpy as np
from flask import render_template, url_for, flash, redirect, request, make_response, jsonify, abort
from flask_wtf import FlaskForm
from wtforms import StringField, TextAreaField, SubmitField
from wtforms.validators import DataRequired

import json
from werkzeug.utils import secure_filename

# Loading raw data and clean it
f = open("market_data_cleaned.pickle", "rb")
market_data_cleaned = pickle.load(f)

f = open("macro_df.pickle", "rb")
macro_df = pickle.load(f)

f = open("fff_data_cleaned.pickle", "rb")
fff_data_cleaned = pickle.load(f)

date = '2008-09'

def get_average_sentiment(market_data, date):
    df_senti = market_data
    state_num = df_senti.iloc[(df_senti.loc[df_senti.Date == date].index).tolist()[0]].Score_Statement
    min_num = df_senti.iloc[(df_senti.loc[df_senti.Date == date].index).tolist()[0]].Score_Minutes
    news_num = df_senti.iloc[(df_senti.loc[df_senti.Date == date].index).tolist()[0]].Score_News
    
    return ((state_num+min_num+news_num) / 3)


def make_home_plot(date = date):
    
            
    app = dash.Dash()

    # Initialize figure with subplots
    # fig = make_subplots(
    # rows=2, cols=3, subplot_titles=("Plot 1", "Plot 2", "Plot 3", "Plot 4", "Plot 5")
    # )

    """
    Market Average Sentiment
    """
    avg = get_average_sentiment(market_data_cleaned, date).round(4)

    word = ""
    if avg > 0:
        word = "Overall Hawkish"
    elif avg < 0:
        word = "Overall Dovish"
    elif avg == 0:
        word = "Overall Neutral"
    
    fig1 = go.Figure()
    fig1.add_trace(go.Indicator(
        title={
            'text': "Average Sentiment Score for " + pd.to_datetime(date).strftime("%B %Y"), 'font.size': 20},
        mode = "delta",
        delta = {'reference': 0, 'font.size': 1},
        domain = {'row': 0, 'column': 0}))
    
    fig1.add_trace(go.Indicator(
        mode = "number",
        number={"font":{"size":80}},
        value = avg,
        domain = {'row': 1, 'column': 0}))
    
    fig1.add_trace(go.Indicator(
        title = {'text': "<"+word+">", 
                 'font.size': 20,
                 'font.family': 'Courier New Bold',
                 'font.color':'#401664',            
                 'align': 'center'},
        mode = 'delta',
        delta = {'reference': 0, 'font.size': 1},
        domain = {'row': 2, 'column': 0}))
    
    
    fig1.update_layout(
        grid = {'rows': 3, 'columns': 1, 'pattern': "independent"},
        paper_bgcolor = "white", 
        font_family="Times New Roman Bold",
        font_color="black",
       
        margin=dict(l=8, r=8, t=25, b=5),
        autosize=True
        )
    fig1.update_xaxes(automargin=True)

    """
    Sentiment Score
    """
    #currently is static and hard coded. 
    # need to implment date picker for entire dashboard for home page
    df_senti = market_data_cleaned
    fig2 = go.Figure()
    #Statement
    fig2.add_trace(go.Indicator(
        mode = "number+delta",
        value = (df_senti.iloc[(df_senti.loc[df_senti.Date == date].index).tolist()[0]].Score_Statement).round(4),
        number={"font":{"size":40}},
        delta = {'position': "right", 'reference': (df_senti.iloc[(df_senti.loc[df_senti.Date == date].index-1).tolist()[0]].Score_Statement).round(4), "font":{"size":20}},
        title = {'text':"FOMC STATEMENT SENTIMENTS SCORE"+'<br>'+ '='*len("FOMC Statement Sentiments Score"), 
                 'font.size': 15, 
                 'font.color': '#401664', 
                 'font.family':'Courier New'},
        domain = {'row': 1, 'column': 0}))
    
    fig2.add_trace(go.Indicator(
        title = {'text': "<"+df_senti.iloc[(df_senti.loc[df_senti.Date == date].index).tolist()[0]].Statement_Sentiments+">", 
                 'font.size': 17,
                 'font.family': 'Courier New', 
                 'font.color': '#401664',
                 'align': 'right'},
        mode = 'delta',
        delta = {'reference': 0, 'font.size': 1},
        domain = {'row': 1, 'column': 1}))

    #Minutes
    fig2.add_trace(go.Indicator(
        mode = "number+delta",
        value = (df_senti.iloc[(df_senti.loc[df_senti.Date == date].index).tolist()[0]].Score_Minutes).round(4),
        number={"font":{"size":40}},
        delta = {'position': "right", 'reference': (df_senti.iloc[(df_senti.loc[df_senti.Date == date].index-1).tolist()[0]].Score_Minutes).round(4), "font":{"size":20}},
        title = {'text':"FOMC MINUTES SENTIMENTS SCORE"+'<br>'+ '='*len("FOMC Minutes Sentiments Score"),
                 'font.size': 15, 
                 'font.color': '#401664', 
                 'font.family':'Courier New'},
        domain = {'row': 3, 'column': 0}))
    
    fig2.add_trace(go.Indicator(
        title = {'text': "<"+df_senti.iloc[(df_senti.loc[df_senti.Date == date].index).tolist()[0]].Minutes_Sentiments+">", 
                 'font.size': 17,
                 'font.color': '#401664',
                 'font.family': 'Courier New', 
                 'align': 'right'},
        mode = 'delta',
        delta = {'reference': 0, 'font.size': 1},
        domain = {'row': 3, 'column': 1}))

    #News
    fig2.add_trace(go.Indicator(
        mode = "number+delta",
        value = (df_senti.iloc[(df_senti.loc[df_senti.Date == date].index).tolist()[0]].Score_News).round(4),
        number={"font":{"size":40}},
        delta = {'position': "right", 'reference': (df_senti.iloc[(df_senti.loc[df_senti.Date == date].index-1).tolist()[0]].Score_News).round(4), "font":{"size":20}},
        title = {'text':"NEWS SENTIMENTS SCORE"+'<br>'+ '='*len("News Sentiments Score"), 
                 'font.size': 15, 
                 'font.color': '#401664', 
                 'font.family':'Courier New'},
        domain = {'row': 5, 'column': 0}))
    
    fig2.add_trace(go.Indicator(
        title = {'text': "<"+df_senti.iloc[(df_senti.loc[df_senti.Date == date].index).tolist()[0]].News_Sentiments+">", 
                 'font.size': 17,
                 'font.color': '#401664',
                 'font.family': 'Courier New',
                 'align': 'right'},
        mode = 'delta',
        delta = {'reference': 0, 'font.size': 1},
        domain = {'row': 5, 'column': 1}))

    
    fig2.update_layout(
        grid = {'rows': 6, 'columns': 2, 'pattern': "independent"},
        paper_bgcolor = "white", 
        font_family="Times New Roman Bold",
        font_color="black",
        title={
            'text': "Breakdown of Sentiment Scores for " + pd.to_datetime(date).strftime("%B %Y"),
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top', 
            'font.size':20},
        margin=dict(l=150, r=100, t=75, b=20),
        autosize=True,)
    fig2.update_xaxes(automargin=True)


    """"
    Macroeconomic Indicators Contributions Plot
    """
    ## Setting up values
    #T10Y3M
    df_plot = macro_df
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
    colors = ['#401664', '#D71C2B', '#EE2033', '#F78E99', '#FBC9CF', 'lavender']

    fig3 = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.5, 
                                 textfont = {'family':'Courier New', 'color':'black'},
                                 textfont_size=13,
                                 showlegend=True, 
                                 marker = dict(colors=colors,line=dict(color='#000000', width=0.4)))])
    fig3.update_traces(textposition='outside', textinfo='percent')
    fig3.update_layout(
        paper_bgcolor = "white",
        font_color="black",
        title={
            'text': "Indicators Contributions for " + pd.to_datetime(date).strftime("%B %Y"),
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top', 
            'font.size':20,
            'font.family': 'Times New Roman Bold'},
        margin=dict(l=100, r=100, t=100, b=100),)

    """"
    Macroeconomic Values Indication
    """
    #date= '2004-09-01'
    ## Setting up values
    #T10Y3M
    df_plot = macro_df
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
    fig4 = go.Figure()


    #T10Y3M Indicator
    fig4.add_trace(go.Indicator(
        mode = "number+delta",
        value = abs(value_T10Y3M.round(4)),
        number={"font":{"size":40}},
        delta = {'position': "right", 'reference':abs(value_T10Y3M_prev.round(4)), "font":{"size":20}},
        title = {'text':"BOND YIELD SPREAD"+'<br>'+ '='*len("Bond Yield Spread"), 
                 'font.size': 15, 
                 'font.color': '#401664', 
                 'font.family':'Courier New'},
        domain = {'row': 0, 'column': 0}))

    #EMRATIO Indicator
    fig4.add_trace(go.Indicator(
        mode = "number+delta",
        value = abs(value_EMRATIO.round(4)),
        number={"font":{"size":40}},
        delta = {'position': "right", 'reference':abs(value_EMRATIO_prev.round(4)), "font":{"size":20}},
        title = {'text':"EMPLOYMENT"+'<br>'+ '='*len("Employment"), 
                 'font.size': 15,
                 'font.color': '#401664', 
                 'font.family':'Courier New'},
        domain = {'row': 0, 'column': 1}))

    #GDP Indicator
    fig4.add_trace(go.Indicator(
        mode = "number+delta",
        value = abs(value_GDPC1.round(4)),
        number={"font":{"size":40}},
        delta = {'position': "right", 'reference':abs(value_GDPC1_prev.round(4)), "font":{"size":20}},
        title = {'text':"DOMESTIC OUTPUT"+'<br>'+ '='*len("Domestic Output"), 
                 'font.size': 15,
                 'font.color': '#401664', 
                 'font.family':'Courier New'},
        domain = {'row': 1, 'column': 0}))

    #MEDCPI Indicator
    fig4.add_trace(go.Indicator(
        mode = "number+delta",
        value = abs(value_MEDCPI.round(4)),
        number={"font":{"size":40}},
        delta = {'position': "right", 'reference':abs(value_MEDCPI_prev.round(4)), "font":{"size":20}},
        title = {'text':"INFLATION"+'<br>'+ '='*len("Inflation"), 
                 'font.size': 15,
                 'font.color': '#401664', 
                 'font.family':'Courier New'},
        domain = {'row': 1, 'column': 1}))

    #HD index Indicator
    fig4.add_trace(go.Indicator(
        mode = "number+delta",
        value = abs(value_HD_index.round(4)),
        number={"font":{"size":40}},
        delta = {'position': "right", 'reference':abs(value_HD_index_prev.round(4)), "font":{"size":20}},
        title = {'text':"HAWKISH-DOVISH INDEX"+'<br>'+ '='*len("Hawisk-Dovish Index"), 
                 'font.size': 15,
                 'font.color': '#401664', 
                 'font.family':'Courier New'},
        domain = {'row': 2, 'column': 0}))

    #shifted target Indicator
    fig4.add_trace(go.Indicator(
        mode = "number+delta",
        value = abs(value_shifted_target.round(4)),
        number={"font":{"size":40}},
        delta = {'position': "right", 'reference':abs(value_shifted_target_prev.round(4)), "font":{"size":20}},
        title = {'text':"PREVIOUS MONTH RATE"+'<br>'+ '='*len("Previous Month Rate"), 
                 'font.size': 15,
                 'font.color': '#401664', 
                 'font.family':'Courier New'},
        domain = {'row': 2, 'column': 1}))

    #Configure layout
    fig4.update_layout(
        grid = {'rows': 3, 'columns': 2, 'pattern': "independent"},
        paper_bgcolor = "white", 
        font_family="Times New Roman Bold",
        font_color="black",
        # width=590, height=470,
        title={
            'text': "Contribution of Indicators for " + pd.to_datetime(date).strftime("%B %Y"),
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top', 
            'font.size': 20},
        margin=dict(l=50, r=50, t=100, b=15),
        autosize=True)
    fig4.update_xaxes(automargin=True)


    """"
    Fed Funds Futures
    """
    date = '2022-09-20'
    df_fff =fff_data_cleaned
    #transform df
    new_df_fff = df_fff.melt(id_vars=["Date"],
                             var_name="Basis Points",
                             value_name="Probability")
    fig5 = px.bar(new_df_fff, x='Basis Points', y='Probability', animation_frame='Date', 
             color_discrete_sequence =['#401664']*len(new_df_fff))
    
    fig5.update_layout(
        font_family="Courier New",
        font_color="black",
        title_font_family="Times New Roman Bold",
        title_font_color="black",
        title_text='Probability of Change in Target Rates by Basis Points', 
        title_x=0.5, 
        title_font_size = 20,
        
        xaxis_title="",
        yaxis_title="Probability of Change", 
        plot_bgcolor = 'white',
        autosize=True
    )
    
    fig5.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#ECECEC', zeroline=True, zerolinecolor='lightgrey', automargin=True)
    fig5.update_yaxes(range=[-0.1, 1.1],showgrid=True, gridwidth=1, gridcolor='#ECECEC', zeroline=True, zerolinecolor='lightgrey')

    app.layout = html.Div([
                    #first
                    html.P([
                        dcc.Graph(figure=fig1),
                        dcc.Graph(figure=fig2)],
                        style = {'width' : '100%',
                                'max-width': '100%',
                                'fontSize' : '20px',
                                # 'padding-left' : '100px',
                                'margin': 0,
                                'display': 'inline-block'},),  
                    #second
                    html.P([
                        dcc.Graph(figure=fig3),
                        dcc.Graph(figure=fig4)],
                        style = {'width' : '100%',
                                'max-width': '100%',
                                'fontSize' : '20px',
                                # 'padding-left' : '100px',
                                'margin': 0,
                                'display': 'inline-block'},),  
                    #third
                    html.P([
                        dcc.Graph(figure=fig5)],
                        style = {'width' : '100%',
                                'max-width': '100%',
                                'fontSize' : '20px',
                                # 'padding-left' : '100px',
                                'margin': 0,
                                'display': 'inline-block'},),
                    
    ], style = {'display': 'flex', 'flex-direction': 'row', 'width' : "100%"}  )
    return app

    

if __name__ == '__main__':
    app = make_home_plot()
    app.run_server()