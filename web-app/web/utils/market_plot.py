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
import dash
import dash as dcc
import dash as html
from web.utils.utils import load_market_data

## Time series of market sentiments (drill down)

def display_market_sentiments_drill_down_1(market_data):
    df_senti=market_data
    #Statement
    fig_statement = px.line(df_senti, x='Date', y='Score_Statement',
                            labels={"Score_Statement": "FOMC Statement Sentiments Score"})

    fig_statement.update_layout(
        font_family="Courier New",
        font_color="black",
        title_font_family="Times New Roman Bold",
        title_font_color="black",
        title_text='Time Series of FOMC Statements Sentiments', 
        title_x=0.5
    )

    fig_statement.update_xaxes(rangeslider_visible=True)
    fig_statement.update_yaxes(range=[-1.1, 1.1])
    
    plot_json = json.dumps(fig_statement, cls=plotly.utils.PlotlyJSONEncoder)
    return plot_json

def display_market_sentiments_drill_down_2(market_data):
    df_senti=market_data
    #Minutes
    fig_minutes = px.line(df_senti, x='Date', y='Score_Minutes',
                            labels={"Score_Minutes": "FOMC Minutes Sentiments Score"})

    fig_minutes.update_layout(
        font_family="Courier New",
        font_color="black",
        title_font_family="Times New Roman Bold",
        title_font_color="black",
        title_text='Time Series of FOMC Minutes Sentiments', 
        title_x=0.5
    )

    fig_minutes.update_xaxes(rangeslider_visible=True)
    fig_minutes.update_yaxes(range=[-1.1, 1.1])

    plot_json = json.dumps(fig_minutes, cls=plotly.utils.PlotlyJSONEncoder)
    return plot_json

def display_market_sentiments_drill_down_3(market_data):

    df_senti=market_data
    #News
    fig_news = px.line(df_senti, x='Date', y='Score_News',
                            labels={"Score_News": "News Sentiments Score"})

    fig_news.update_layout(
        font_family="Courier New",
        font_color="black",
        title_font_family="Times New Roman Bold",
        title_font_color="black",
        title_text='Time Series of News Sentiments', 
        title_x=0.5
    )

    fig_news.update_xaxes(rangeslider_visible=True)
    fig_news.update_yaxes(range=[-1.1, 1.1])

    plot_json = json.dumps(fig_news, cls=plotly.utils.PlotlyJSONEncoder)
    return plot_json


    '''
    Placeholder function for plots
    '''
    df = market_data
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
        #width=500, height=400
    )
    fig.update_layout(width=500, height=400)
    plot_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return plot_json

