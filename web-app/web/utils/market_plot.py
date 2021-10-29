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
from sklearn.feature_extraction.text import CountVectorizer

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

    fig_statement.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        )
    )
    
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
    
    fig_minutes.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=1, label="1m", step="month", stepmode="backward"),
            dict(count=6, label="6m", step="month", stepmode="backward"),
            dict(count=1, label="YTD", step="year", stepmode="todate"),
            dict(count=1, label="1y", step="year", stepmode="backward"),
            dict(step="all")
        ])
    )
)

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

    fig_news.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        )
    )
    
    plot_json = json.dumps(fig_news, cls=plotly.utils.PlotlyJSONEncoder)
    return plot_json


def plot_top_n_trigram(data):
    corpus = data
    n=3
    vec = CountVectorizer(ngram_range=(n, n)).fit([corpus])
    bag_of_words = vec.transform([corpus])
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    df_plot = pd.DataFrame(words_freq[:10], columns = ['Word/Phrase' , 'Count'])
    fig = px.bar(df_plot, x='Word/Phrase', y='Count')
    
    fig.update_layout(
        font_family="Courier New",
        font_color="black",
        title_font_family="Times New Roman Bold",
        title_font_color="black",
        title_text='Top 10 Most Common Words/Phrases', 
        title_x=0.5
    )
    plot_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return plot_json
