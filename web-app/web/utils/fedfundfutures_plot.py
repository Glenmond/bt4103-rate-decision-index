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
from web.utils.utils import load_fff_data

def plot_fff_results(data):
    '''
    Placeholder function for plots
    '''
    fff_data = data.copy()
    fff_data['Date'] = pd.to_datetime(fff_data['Date'])
    titles = ["Model Results"]  + list(y.strftime("%d %B %Y FOMC Meeting") for y in fff_data['Date'])

    fig = make_subplots(
        rows=6, cols=2,
        shared_xaxes=True,
        subplot_titles = titles,
        vertical_spacing=0.03,
        horizontal_spacing=0.3,
        specs=[[{"type": "table", "colspan": 2}, None],
            [{"type": "bar"},{"type": "bar"}],
            [{"type": "bar"},{"type": "bar"}],
            [{"type": "bar"},{"type": "bar"}],
            [{"type": "bar"},{"type": "bar"}],
            [{"type": "bar"},{"type": "bar"}]]
    )
    #table_df = fff_data.copy().round(2).reset_index()
    table_df = fff_data.copy().round(2)
    table_df['Date'] = pd.to_datetime(table_df['Date']).apply(lambda x: x.strftime("%Y-%m-%d"))
    fig.add_trace(
        go.Table(
            header=dict(
                values=["Date", "0-25<br>BPS", "25-50<br>BPS",
                        "50-75<br>BPS", "75-100<br>BPS", "100-125<br>BPS",
                        "125-150<br>BPS", "150-175<br>BPS", "175-200<br>BPS"],
                font=dict(size=10),
                align="left"
            ),
            cells=dict(
                values=[table_df[k].tolist() for k in table_df.columns],
                align = "left")
        ),
        row=1, col=1
    )

    rows = 2
    cols = 1
    df = fff_data.copy().round(3)
    for dt in df.index:
        monthdf = df.loc[[dt]].T
        monthdf.columns = ['value']
        fig.add_trace(
            go.Bar(
                y = monthdf.index,
                x = monthdf.value,
                text = monthdf.value,
                name = str(dt),
                showlegend=True,
                orientation="h",  
            ),
            row=rows, col=cols,
        )
        if cols == 1:
            cols+=1
        elif cols == 2:
            cols = 1
            rows +=1

    fig.update_layout(
        height=2000,
        showlegend=False,
        title_text="Target Rate Predictions for FOMC Meetings",
    )
    #fig.update_layout(width=1500, height=500)
    plot_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return plot_json


def plot_futures_pred_vs_fomc(predictions, fomc):
    low, mid, high = fomc
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=low['Date'], y=low['Value'],
                        line = dict(color='royalblue', width=4, dash='dashdot'),
                        line_shape='spline',
                        name='Low'))
    fig.add_trace(go.Scatter(x=mid['Date'], y=mid['Value'],
                        line = dict(color='royalblue', width=4),
                        name='Midpoint'))
    fig.add_trace(go.Scatter(x=high['Date'], y=high['Value'],
                        line = dict(color='royalblue', width=4, dash='dashdot'),
                        name='High'))

    fig.add_trace(go.Scatter(x=predictions['Date'], y=predictions['Prediction'], 
                            mode='lines+markers', name='Predicted Rate'))

    annotations = []
    annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.05,
                                xanchor='left', yanchor='bottom',
                                text='Futures Predicted Rate vs FOMC Economic Projections',
                                font=dict(family='Arial',
                                            size=15,
                                            color='rgb(37,37,37)'),
                                showarrow=False))
    annotations.append(dict(xref='paper', yref='paper', x=0.5, y=-0.1,
                                xanchor='center', yanchor='top',
                                text='Source: FRED: FOMC Summary of Economic Projections for the Fed Funds Rate, Range',
                                font=dict(family='Arial',
                                            size=14,
                                            color='rgb(150,150,150)'),
                                showarrow=False))


    fig.update_layout(annotations=annotations,  plot_bgcolor='white')
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')
    plot_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return plot_json
