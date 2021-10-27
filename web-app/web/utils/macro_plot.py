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
    plot_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return plot_json