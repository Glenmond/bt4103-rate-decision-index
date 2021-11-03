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
from web.macro_model_2.macroeconomics_model import MacroModel, MacroData

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


def plot_main_plot(data):
    macro_data = MacroData(data)
    macro_model = MacroModel(macro_data)

    macro_model.fit_data()

    #macro_model.assess_val_set_performance()
    #macro_model.assess_test_set_performance()

    fig = go.Figure()

    # training data
    training_data = macro_data.X_train.copy().append(macro_data.X_val).sort_index()
    training_data_y = macro_data.y_train.copy().append(macro_data.y_val).sort_index()

    y_train_to_plot = pd.DataFrame(training_data_y.values, index=training_data_y.index.strftime('%Y-%m'), columns=['Federal Funds Rate'])
    fig.add_trace(go.Scatter(
                    y=y_train_to_plot['Federal Funds Rate'], x=y_train_to_plot.index, 
                    mode='lines', name='Training Data'))

    y_perf_res, y_perf = macro_model.predict(training_data)
    y_perf_to_plot = pd.DataFrame(y_perf,index=training_data.index.strftime('%Y-%m'), columns=['Federal Funds Rate'])
    fig.add_trace(go.Scatter(
                    y=y_perf_to_plot['Federal Funds Rate'], x=y_perf_to_plot.index, 
                    mode='lines', name='Model Performance on Training Data'))

    y_test_to_plot = pd.DataFrame(macro_data.y_test.values,index=macro_data.y_test.index.strftime('%Y-%m'), columns=['Federal Funds Rate'])
    fig.add_trace(go.Scatter(
                    y=y_test_to_plot['Federal Funds Rate'], x=y_test_to_plot.index, 
                    mode='lines', name='Testing Data'))

    y_pred_res, y_pred = macro_model.predict(macro_data.X_test)
    y_pred_to_plot = pd.DataFrame(y_pred,index=macro_data.y_test.index.strftime('%Y-%m'), columns=['Federal Funds Rate'])
    fig.add_trace(go.Scatter(
                    y=y_pred_to_plot['Federal Funds Rate'], x=y_pred_to_plot.index, 
                    mode='lines', name='Model Performance on Testing Data'))

    conf_int_95 = y_pred_res.summary_frame(0.05)
    conf_int_95_upper = conf_int_95['mean_ci_upper']
    conf_int_95_lower = conf_int_95['mean_ci_lower']
    fig.add_trace(go.Scatter(
                    y = conf_int_95_lower, x=y_test_to_plot.index,
                    fill=None, line_color='rgba(255, 0, 0, 0.1)', showlegend=False))
    fig.add_trace(go.Scatter(
                    y = conf_int_95_upper, x=y_test_to_plot.index,
                    fill='tonexty', line_color='rgba(255, 0, 0, 0.1)', 
                    fillcolor='rgba(255, 0, 0, 0.1)', name='95% Confidence Interval'))


    conf_int_99 = y_pred_res.summary_frame(0.01)
    conf_int_99_upper = conf_int_99['mean_ci_upper']
    conf_int_99_lower = conf_int_99['mean_ci_lower']
    fig.add_trace(go.Scatter(
                    y = conf_int_99_lower, x=y_test_to_plot.index,
                    fill=None, line_color='rgba(0, 0, 255, 0.1)', showlegend=False))
    fig.add_trace(go.Scatter(
                    y = conf_int_99_upper, x=y_test_to_plot.index,
                    fill='tonexty', line_color='rgba(0, 0, 255, 0.1)', 
                    fillcolor='rgba(0, 0, 255, 0.1)', name='99% Confidence Interval'))


    fig.update_layout(title='Prediction of Federal Funds Rate',
        xaxis_title='Date',
        yaxis_title='Federal Funds Rate')

    plot_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return plot_json



