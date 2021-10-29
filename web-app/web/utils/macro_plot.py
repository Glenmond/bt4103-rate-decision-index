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
from web.utils.utils import load_macro_data

def plot_gdp_index(gdp_sub_index):
    x = gdp_sub_index['Date']

    fig = go.Figure()

    fig.add_trace(go.Line(x=x, y=gdp_sub_index['GDPC1'],name='GDP', marker=dict(color="Black"))),
    fig.add_trace(go.Line(x=x, y=gdp_sub_index['PCEC96'],name='Domestic Consumption', marker=dict(color="lightcoral"))),
    fig.add_trace(go.Line(x=x, y=gdp_sub_index['GPDIC1'], name='Domestic Investment', marker=dict(color="sandybrown")))
    fig.add_trace(go.Line(x=x, y=gdp_sub_index['GCEC1'], name='Government Expenditure', marker=dict(color="darkseagreen")))
    fig.add_trace(go.Line(x=x, y=gdp_sub_index['NETEXC'], name='Net Export', marker=dict(color="cornflowerblue")))

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
            direction="down",
            pad={"r": 5, "t": 5},
            showactive=True,
            x=1.3,
            xanchor="left",
            y=1.2,
            yanchor="top"
            )
        ],

    )
    #fig.update_layout(width=1500, height=500)
    fig.update_xaxes(rangeslider_visible=True)
    plot_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return plot_json



def plot_employment_index(gdp_sub_index):
    dfEI = gdp_sub_index
    x = gdp_sub_index['Date']

    fig = go.Figure()

    fig.add_trace(go.Line(x=x, y=dfEI['PAYEMS'],name='PAYEMS', marker=dict(color="Black"))),
    fig.add_trace(go.Line(x=x, y=dfEI['USPRIV'],name='Private', marker=dict(color="lightcoral"))),
    fig.add_trace(go.Line(x=x, y=dfEI['CES9091000001'], name='Federal', marker=dict(color="sandybrown")))
    fig.add_trace(go.Line(x=x, y=dfEI['USCONS'], name='Construction', marker=dict(color="lightgreen")))
    fig.add_trace(go.Line(x=x, y=dfEI['MANEMP'], name='Manufacturing', marker=dict(color="cornflowerblue")))

    fig.update_layout(title_text='Employment and its components',
                    xaxis_title='Date', yaxis_title='Value')


    fig.update_layout(
        updatemenus=[
            dict(active=0,
                buttons=list([
                dict(label="All",
                    method="update",
                    args=[{"visible":[True,True,True,True,True]},
                        {"title":"ALL"}]),
                dict(label="USPRIV",
                    method="update",
                    args=[{"visible":[True, True, False,False,False]},
                        {"title":"Private"}]),
                dict(label="CES9091000001",
                    method="update",
                    args=[{"visible":[True,False,True,False,False]},
                        {"title":"Federal"}]),
                dict(label="USCONS",
                    method="update",
                    args=[{"visible":[True,False,False,True,False]},
                        {"title":"Construction"}]),
                dict(label="MANEMP",
                    method="update",
                    args=[{"visible":[True,False,False,False,True]},
                        {"title":"Manufacturing"}])
            ]),
            direction="down",
            pad={"r": 5, "t": 5},
            showactive=True,
            x=1.3,
            xanchor="left",
            y=1.2,
            yanchor="top"
            )
        ]
    )
    fig.update_xaxes(rangeslider_visible=True)
    plot_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return plot_json



def plot_inflation_index(gdp_sub_index):
    df_inflation = gdp_sub_index
    x = df_inflation['Date']

    fig = go.Figure()

    fig.add_trace(go.Line(x=x, y=df_inflation['CPIAUCSL'],name='CPI', marker=dict(color="Black"))),
    fig.add_trace(go.Line(x=x, y=df_inflation['CPIFABSL'],name='Food and Beverage', marker=dict(color="lightcoral"))),
    fig.add_trace(go.Line(x=x, y=df_inflation['CPIAPPSL'], name='Apparel', marker=dict(color="sandybrown")))
    fig.add_trace(go.Line(x=x, y=df_inflation['CPIMEDSL'], name='Medical', marker=dict(color="lightgreen")))
    fig.add_trace(go.Line(x=x, y=df_inflation['CPIHOSSL'], name='Housing', marker=dict(color="cornflowerblue")))
    fig.add_trace(go.Line(x=x, y=df_inflation['CPITRNSL'], name='Transportation', marker=dict(color="thistle")))
    fig.add_trace(go.Line(x=x, y=df_inflation['CPIEDUSL'], name='Education and Communication', marker=dict(color="mediumaquamarine")))
    fig.add_trace(go.Line(x=x, y=df_inflation['CPIRECSL'], name='Recreation', marker=dict(color="darkseagreen")))
    fig.add_trace(go.Line(x=x, y=df_inflation['CPIOGSSL'], name='Other goods and services', marker=dict(color="lightpink")))

    fig.update_layout(title_text='CPI and its components',
                    xaxis_title='Date', yaxis_title='Value')

    fig.update_layout(
        updatemenus=[
            dict(active=0,
                buttons=list([
                dict(label="All",
                    method="update",
                    args=[{"visible":[True,True,True,True,True,True,True,True]},
                        {"title":"ALL"}]),
                dict(label="CPIFABSL",
                    method="update",
                    args=[{"visible":[True,True,False,False,False,False,False,False,False]},
                        {"title":"Food and Beverage"}]),
                dict(label="CPIAPPSL",
                    method="update",
                    args=[{"visible":[True,False,True,False,False,False,False,False,False]},
                        {"title":"Apparel"}]),
                dict(label="CPIMEDSL",
                    method="update",
                    args=[{"visible":[True,False,False,True,False,False,False,False,False]},
                        {"title":"Medical Services"}]),
                    dict(label="CPIHOSSL",
                    method="update",
                    args=[{"visible":[True,False,False,False,True,False,False,False,False]},
                        {"title":"Housing"}]),
                dict(label="CPITRNSL",
                    method="update",
                    args=[{"visible":[True,False,False,False,False,True,False,False,False]},
                        {"title":"Transportation"}]),
                dict(label="CPIEDUSL",
                    method="update",
                    args=[{"visible":[True,False,False,False,False,False,True,False,False]},
                        {"title":"Education and Communication"}]),
                dict(label="CPIRECSL",
                    method="update",
                    args=[{"visible":[True,False,False,False,False,False,False,True,False]},
                        {"title":"Recreation"}]),
                dict(label="CPIOGSSL",
                    method="update",
                    args=[{"visible":[True,False,False,False,False,False,False,False,True]},
                        {"title":"Other goods and services"}])
                


                
                ]),
            direction="down",
            pad={"r": 5, "t": 5},
            showactive=True,
            x=1.3,
            xanchor="left",
            y=1.2,
            yanchor="top"
            )
        ]
    )

    fig.update_xaxes(rangeslider_visible=True)
    plot_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return plot_json