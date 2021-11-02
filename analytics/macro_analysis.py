# this is where we do our main analysis using our models and data
import numpy as np
import pickle

import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output

import pandas as pd
import plotly.graph_objs as go

from data import fetch_data
from macro_model import MacroModel, MacroData

# Get the data
try: # so that we do not need the FRED API call every time we try to access
    data = pd.read_pickle('./macro_model/macro_data_pickle')
except Exception:   
    data = fetch_data()
    data.to_pickle('./macro_model/macro_data_pickle')

macro_data = MacroData(data)
macro_model = MacroModel(macro_data)

macro_model.fit_data()

#macro_model.assess_val_set_performance()
#macro_model.assess_test_set_performance()

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


app.layout = html.Div([
                dcc.Graph(id='plot', figure=fig)
])

if __name__ == '__main__':
    app.run_server(debug=True)
    

    


    
