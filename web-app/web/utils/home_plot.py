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
from web.utils.utils import load_home_data, load_market_data, load_fff_data


def plot_gauge():
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = 0.654,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Rate Hike-Cut", 'font': {'size': 20}},
        delta = {'reference': 0.5, 'increasing': {'color': "mediumseagreen"}},
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

    #fig.update_layout(font = {'color': "darkblue", 'family': "Arial"})
    #fig.update_layout(width=800, height=450)
    plot_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    return plot_json


def plot_market(market_data):
    #currently is static and hard coded. 
    # need to implment date picker for entire dashboard for home page
    date = '2000-04-30'
    df_senti = market_data
    
    layout = go.Layout(
        margin=go.layout.Margin(
        l=150, #left margin
        r=0, #right margin
        b=30, #bottom margin
        t=30  #top margin
        )
    )
    
    fig = go.Figure(layout=layout)
    
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
        font_color="black",
        #template='plotly_dark'
        )
    
    fig.update_layout(width=590, height=450)
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
    
    avg = round(get_average_sentiment(market_data),4)
    
    #avg = get_average_sentiment(market_data)
    
    word = ""
    if avg > 0:
        word = "Overall Hawkish"
    elif avg < 0:
        word = "Overall Dovish"
    elif avg == 0:
        word = "Overall Neutral"
        
    layout = go.Layout(
        margin=go.layout.Margin(
        l=0, #left margin
        r=100, #right margin
        b=0, #bottom margin
        t=50  #top margin
        )
    )
    
    fig = go.Figure(layout=layout)

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
                 'font.size': 16,
                 'font.family': 'Courier New Bold',
                 'font.color':'saddlebrown'},
        mode = 'delta',
        delta = {'reference': 0, 'font.size': 1},
        domain = {'row': 1, 'column': 0}))
    
    fig.update_layout(
        grid = {'rows': 2, 'columns': 1, 'pattern': "independent"}, 
        paper_bgcolor = "white", 
        font_family="Times New Roman Bold",
        font_color="black")
    

    fig.update_layout(width=600, height=100)
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

def plot_gdp_index(home_data):
    df = home_data
    #df['period'] = df.index
    #df['date'] = pd.PeriodIndex(df['period'], freq='Q').to_timestamp()
    
    x = df['Date']

    fig = go.Figure()

    fig.add_trace(go.Line(x=x, y=df['GDPC1'],name='GDP', marker=dict(color="Black"))),
    fig.add_trace(go.Line(x=x, y=df['PCEC96'],name='Consumption', marker=dict(color="lightcoral"))),
    fig.add_trace(go.Line(x=x, y=df['GPDIC1'], name='Investment', marker=dict(color="sandybrown")))
    fig.add_trace(go.Line(x=x, y=df['GCEC1'], name='Expenditure', marker=dict(color="darkseagreen")))
    fig.add_trace(go.Line(x=x, y=df['NETEXC'], name='Net Export', marker=dict(color="cornflowerblue")))

    fig.update_layout(title_text='Macroeconomic Indicators',
                    xaxis_title='Date', yaxis_title='Value')
    fig.update_layout(
        updatemenus=[
            dict(active=0,
                buttons=list([
                dict(label="All",
                    method="update",
                    args=[{"visible":[True,True,True,True,True]},
                        {"title":"ALL"}]),
                dict(label="Employment",
                    method="update",
                    args=[{"visible":[True, True, False,False,False]},
                        {"title":"Employment"}]),
                dict(label="Median CPI",
                    method="update",
                    args=[{"visible":[True,False,True,False,False]},
                        {"title":"Median CPI"}]),
                dict(label="Real GDP",
                    method="update",
                    args=[{"visible":[True,False,False,True,False]},
                        {"title":"Real GDP"}]),
                dict(label="Bond Yields",
                    method="update",
                    args=[{"visible":[True,False,False,False,True]},
                        {"title":"Bond Yields"}])
            ]),
            direction="down",
            pad={"r": 5, "t": 5},
            showactive=True,
            x=1,
            xanchor="center",
            y=1.2,
            yanchor="bottom"
            )
        ],
    )
    #fig.update_xaxes(rangeslider_visible=True)
    #fig.update_layout(width=600, height=400)
    plot_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return plot_json

def plot_macro_maindashboard(df_plot):
    plot = go.Figure(data=[
        go.Scatter(
        name='actual_values',
        x=df_plot.Date.tolist(),
        y=df_plot.actual_values.tolist(),
        marker_color='#A52A2A' #change color of line
    ),
        go.Scatter(
        name='predicted',
        x=df_plot.Date.tolist(),
        y=df_plot.predicted.tolist(),
        marker_color='#000000' #change color of line
    ),  
        go.Scatter(
        name='T10Y3M',
        x=df_plot.Date.tolist(),
        y=df_plot.T10Y3M.tolist(),
        marker_color='#FA8072' #change color of line
    ),
        go.Scatter(
        name='EMRATIO_MEDWAGES',
        x=df_plot.Date.tolist(),
        y=df_plot.EMRATIO_MEDWAGES.tolist(),
        marker_color='#4682B4' #change color of line
    ),
        go.Scatter(
        name='EMRATIO',
        x=df_plot.Date.tolist(),
        y=df_plot.EMRATIO.tolist(),
        marker_color='#00008B' #change color of line
    ),
        go.Scatter(
        name='GDPC1',
        x=df_plot.Date.tolist(),
        y=df_plot.GDPC1.tolist(),
        marker_color='#008B8B' #change color of line
    ),
        go.Scatter(
        name='MEDCPI',
        x=df_plot.Date.tolist(),
        y=df_plot.MEDCPI.tolist(),
        marker_color='#006400' #change color of line
    ),
        go.Scatter(
        name='MEDCPI_PPIACO',
        x=df_plot.Date.tolist(),
        y=df_plot.MEDCPI_PPIACO.tolist(),
        marker_color='#8B008B' #change color of line
    ),
        go.Scatter(
        name='HD_index',
        x=df_plot.Date.tolist(),
        y=df_plot.HD_index.tolist(),
        marker_color='#FF8C00' #change color of line
    ),
        go.Scatter(
        name='shifted_target',
        x=df_plot.Date.tolist(),
        y=df_plot.shifted_target.tolist(),
        marker_color='#8FBC8F' #change color of line
    )
    ])

    plot.update_layout(
        updatemenus=[
            dict(
                active=0,
                buttons=list([
                    dict(label="All",
                         method="update",
                         args=[{"visible": [True, True, True, True, True, True, True, True, True, True]},
                               {"title": "All Indicators"}]),
                    dict(label="T10Y3M",
                         method="update",
                         args=[{"visible": [True, True, True, False, False, False, False, False, False, False]},
                               {"title": "T10Y3M",
                                }]),
                    dict(label="EMRATIO_MEDWAGES",
                         method="update",
                         args=[{"visible": [True, True, False, True, False, False, False, False, False, False]},
                               {"title": "EMRATIO_MEDWAGES",
                                }]),
                    dict(label="EMRATIO",
                         method="update",
                         args=[{"visible": [True, True, False, False, True, False, False, False, False, False]},
                               {"title": "EMRATIO",
                                }]),
                    dict(label="GDPC1",
                         method="update",
                         args=[{"visible": [True, True, False, False, False, True, False, False, False, False]},
                               {"title": "GDPC1",
                                }]),
                    dict(label="MEDCPI",
                         method="update",
                         args=[{"visible": [True, True, False, False, False, False, True, False, False, False]},
                               {"title": "MEDCPI",
                                }]),
                    dict(label="MEDCPI_PPIACO",
                         method="update",
                         args=[{"visible": [True, True, False, False, False, False, False, True, False, False]},
                               {"title": "MEDCPI_PPIACO",
                                }]),
                    dict(label="HD_index",
                         method="update",
                         args=[{"visible": [True, True, False, False, False, False, False, False, True, False]},
                               {"title": "HD_index",
                                }]),
                    dict(label="shifted_target",
                         method="update",
                         args=[{"visible": [True, True, False, False, False, False, False, False, False, True]},
                               {"title": "shifted_target",
                                }]),
                ]),
            direction="down",
            pad={"r": 6, "t": 5},
            showactive=True,
            x=1.4,
            xanchor="center",
            y=1.2,
            yanchor="bottom"
            )
        ])

    plot.update_layout(
            font_family="Courier New",
            font_color="black",
            title_font_family="Times New Roman Bold",
            title_font_color="black",
            title_text='All Indicators', 
            title_x=0.5
        )
    plot.update_layout(width=625, height=550)
    plot.update_xaxes(rangeslider_visible=True)

    plot_json = json.dumps(plot, cls=plotly.utils.PlotlyJSONEncoder)
    return plot_json

#to remobe
def plot_market_average2(market_data):
    date = '2000-04-30'
    df_senti = market_data
    
    avg = round(get_average_sentiment(market_data),4)
    
    #avg = get_average_sentiment(market_data)
    
    word = ""
    if avg > 0:
        word = "Overall Hawkish"
    elif avg < 0:
        word = "Overall Dovish"
    elif avg == 0:
        word = "Overall Neutral"
        
    layout = go.Layout(
        margin=go.layout.Margin(
        l=0, #left margin
        r=100, #right margin
        b=0, #bottom margin
        t=50  #top margin
        )
    )
    
    fig = go.Figure(layout=layout)

    fig.add_trace(go.Indicator(
        mode = "number",
        value = avg,
        number = {'valueformat':'a'},
        title = {'text': "Macroeconomic Indicators Score for" + " " + date, 
                 'font.size': 25,
                 'font.family': 'Times New Roman Bold'},
        domain = {'row': 0, 'column': 0}))
    
    fig.add_trace(go.Indicator(
        title = {'text': word, 
                 'font.size': 16,
                 'font.family': 'Courier New Bold',
                 'font.color':'saddlebrown'},
        mode = 'delta',
        delta = {'reference': 0, 'font.size': 1},
        domain = {'row': 1, 'column': 0}))
    
    fig.update_layout(
        grid = {'rows': 2, 'columns': 1, 'pattern': "independent"}, 
        paper_bgcolor = "white", 
        font_family="Times New Roman Bold",
        font_color="black")
    

    fig.update_layout(width=600, height=125)
    plot_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return plot_json

#to remove 
def plot_market_average3(market_data):
    date = '2000-04-30'
    df_senti = market_data
    
    avg = round(get_average_sentiment(market_data),4)
    
    #avg = get_average_sentiment(market_data)
    
    word = ""
    if avg > 0:
        word = "Overall Hawkish"
    elif avg < 0:
        word = "Overall Dovish"
    elif avg == 0:
        word = "Overall Neutral"
        
    layout = go.Layout(
        margin=go.layout.Margin(
        l=0, #left margin
        r=100, #right margin
        b=0, #bottom margin
        t=50  #top margin
        )
    )
    
    fig = go.Figure(layout=layout)

    fig.add_trace(go.Indicator(
        mode = "number",
        value = avg,
        number = {'valueformat':'a'},
        title = {'text': "Fed Fund Futures Score for" + " " + date, 
                 'font.size': 25,
                 'font.family': 'Times New Roman Bold'},
        domain = {'row': 0, 'column': 0}))
    
    fig.add_trace(go.Indicator(
        title = {'text': word, 
                 'font.size': 16,
                 'font.family': 'Courier New Bold',
                 'font.color':'saddlebrown'},
        mode = 'delta',
        delta = {'reference': 0, 'font.size': 1},
        domain = {'row': 1, 'column': 0}))
    
    fig.update_layout(
        grid = {'rows': 2, 'columns': 1, 'pattern': "independent"}, 
        paper_bgcolor = "white", 
        font_family="Times New Roman Bold",
        font_color="black")
    

    fig.update_layout(width=600, height=125)
    plot_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return plot_json
 

#to remove
def display_market_sentiments_drill_down_4(market_data):
    df_senti=market_data
    #Statement
    fig_statement = px.line(df_senti, x='Date', y='Score_Statement',
                            labels={"Score_Statement": "Probability of Rate Decision Indicator"})

    fig_statement.update_layout(
        font_family="Courier New",
        font_color="black",
        title_font_family="Times New Roman Bold",
        title_font_color="black",
        title_text='Time Series of Probability of Rate Decision Indicator', 
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

