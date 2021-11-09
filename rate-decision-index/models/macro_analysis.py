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

from extract import fetch_data
from macro_model import MacroModel, MacroData

conf_int_95_line_colour = 'rgba(255, 0, 0, 0.1)'
conf_int_99_line_colour = 'rgba(0, 0, 255, 0.1)'

# Get the data
try: # so that we do not need the FRED API call every time we try to access
    data = pd.read_pickle('./data/macroeconomic_indicators_data/macro_data_pickle')
except Exception:   
    data = fetch_data()
    data.to_pickle('./data/macroeconomic_indicators_data/macro_data_pickle', protocol = 4)

macro_data = MacroData(data)

try:
    with open('./data/macroeconomic_indicators_data/macro_model_pickle' , 'rb') as f:
        macro_model = pickle.load(f)
except Exception:
    macro_model = MacroModel(macro_data)
    macro_model.fit_data()
    with open('./data/macroeconomic_indicators_data/macro_model_pickle', 'wb') as files:
        pickle.dump(macro_model, files, protocol = 4)
        

app = dash.Dash()

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
                fill=None, line_color=conf_int_95_line_colour, showlegend=False))
fig.add_trace(go.Scatter(
                y = conf_int_95_upper, x=y_test_to_plot.index,
                fill='tonexty', line_color=conf_int_95_line_colour, 
                fillcolor=conf_int_95_line_colour, name='95% Confidence Interval'))


conf_int_99 = y_pred_res.summary_frame(0.01)
conf_int_99_upper = conf_int_99['mean_ci_upper']
conf_int_99_lower = conf_int_99['mean_ci_lower']
fig.add_trace(go.Scatter(
                y = conf_int_99_lower, x=y_test_to_plot.index,
                fill=None, line_color=conf_int_99_line_colour, showlegend=False))
fig.add_trace(go.Scatter(
                y = conf_int_99_upper, x=y_test_to_plot.index,
                fill='tonexty', line_color=conf_int_99_line_colour, 
                fillcolor=conf_int_99_line_colour, name='99% Confidence Interval'))


fig.update_layout(title='Prediction of Federal Funds Rate',
    xaxis_title='Date',
    yaxis_title='Federal Funds Rate')


app.layout = html.Div([
                dcc.Graph(id='plot', figure=fig),
                # slider
                html.P([
                    html.Label("Coefficient"),
                    dcc.Slider(id = 'slider',
                                    #marks = {i : i/10 for i in range(0, 10)},
                                    step = 0.01,
                                    min = -2,
                                    max = 2,
                                    value = 1.71,
                                    tooltip={"placement": "bottom", "always_visible": True},),
                    html.P(id='metrics')], 
                style = {'width' : '80%',
                        'fontSize' : '20px',
                        'padding-left' : '100px',
                        'display': 'inline-block'},),                   
])

@app.callback(Output('plot','figure'),
              Output('metrics','children'),
              Input('slider','value'))

def update_figure(input):
    # refit the data with the new coefficient value
    new_model = MacroModel(macro_data, shift_coef = input)
    new_model.fit_data()

    # return the new figure
    fig = go.Figure()

    # same training data plot
    fig.add_trace(go.Scatter(
                y=y_train_to_plot['Federal Funds Rate'], x=y_train_to_plot.index, 
                mode='lines', name='Training Data'))

    # same testing data plot
    y_test_to_plot = pd.DataFrame(macro_data.y_test.values,index=macro_data.y_test.index.strftime('%Y-%m'), columns=['Federal Funds Rate'])
    fig.add_trace(go.Scatter(
                y=y_test_to_plot['Federal Funds Rate'], x=y_test_to_plot.index, 
                mode='lines', name='Testing Data'))

    # new performance on training data
    new_y_perf_res, new_y_perf = new_model.predict(training_data)
    new_y_perf_to_plot = pd.DataFrame(y_perf,index=training_data.index.strftime('%Y-%m'), columns=['Federal Funds Rate'])
    fig.add_trace(go.Scatter(
                y=new_y_perf_to_plot['Federal Funds Rate'], x=new_y_perf_to_plot.index, 
                mode='lines', name='Model Performance on Training Data'))

    # new performance on test data
    new_y_pred_res, new_y_pred = new_model.predict(macro_data.X_test)
    new_y_pred_to_plot = pd.DataFrame(new_y_pred,index=macro_data.y_test.index.strftime('%Y-%m'), columns=['Federal Funds Rate'])
    fig.add_trace(go.Scatter(
                    y=new_y_pred_to_plot['Federal Funds Rate'], x=new_y_pred_to_plot.index, 
                    mode='lines', name='Model Performance on Testing Data'))

    # new confidence intervals
    new_conf_int_95 = new_y_pred_res.summary_frame(0.05)
    new_conf_int_95_upper = new_conf_int_95['mean_ci_upper']
    new_conf_int_95_lower = new_conf_int_95['mean_ci_lower']
    fig.add_trace(go.Scatter(
                    y = new_conf_int_95_lower, x=y_test_to_plot.index,
                    fill=None, line_color=conf_int_95_line_colour, showlegend=False))
    fig.add_trace(go.Scatter(
                    y = new_conf_int_95_upper, x=y_test_to_plot.index,
                    fill='tonexty', line_color=conf_int_95_line_colour, 
                    fillcolor=conf_int_95_line_colour, name='95% Confidence Interval'))

    new_conf_int_99 = new_y_pred_res.summary_frame(0.01)
    new_conf_int_99_upper = new_conf_int_99['mean_ci_upper']
    new_conf_int_99_lower = new_conf_int_99['mean_ci_lower']
    fig.add_trace(go.Scatter(
                    y = new_conf_int_99_lower, x=y_test_to_plot.index,
                    fill=None, line_color=conf_int_99_line_colour, showlegend=False))
    fig.add_trace(go.Scatter(
                    y = new_conf_int_99_upper, x=y_test_to_plot.index,
                    fill='tonexty', line_color=conf_int_99_line_colour, 
                    fillcolor=conf_int_99_line_colour, name='99% Confidence Interval'))

    fig.update_layout(title='Prediction of Federal Funds Rate',
                        xaxis_title='Date',
                        yaxis_title='Federal Funds Rate')

    # calculate R2 and RMSE
    r2 = r2_score(macro_data.y_test, new_y_pred)
    rmse = math.sqrt(mean_squared_error(macro_data.y_test, new_y_pred))

    metrics_report = f"R2: {r2:.3f}, RMSE: {rmse:.3f}"
    return fig ,metrics_report


if __name__ == '__main__':
    app.run_server(debug=True)