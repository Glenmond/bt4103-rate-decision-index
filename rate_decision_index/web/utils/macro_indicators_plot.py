import numpy as np
import pickle
import math
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd

import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import plotly.graph_objs as go

from models.macro_model import MacroModel, MacroData
from models import update_saved_data

# Configure html for dashboard
html_layout = """
    {%app_entry%}
    <footer>
        {%config%}
        {%scripts%}
        {%renderer%}
    </footer>
"""

conf_int_95_line_colour = 'rgba(255, 0, 0, 0.1)'
conf_int_99_line_colour = 'rgba(0, 0, 255, 0.1)'

def init_dashboard(server, path_to_pickle_files = './models/data/macroeconomic_indicators_data/'):
    """
    Initialize the dashboard when we run the app
    """
    try: # so that we do not need the FRED API call every time we try to access
        data = pd.read_pickle(path_to_pickle_files + 'macro_data_pickle')
        macro_data = MacroData(data, path_to_HD_pickle='./models/data/sentiment_data/historical/')
        with open(path_to_pickle_files + 'macro_model_pickle' , 'rb') as f:
            macro_model = pickle.load(f)
    except Exception:   
        update_saved_data(path_to_folder= path_to_pickle_files, path_to_HD_folder = './models/data/sentiment_data/historical/')
        data = pd.read_pickle(path_to_pickle_files + 'macro_data_pickle')
        macro_data = MacroData(data, path_to_HD_pickle='./models/data/sentiment_data/historical/')
        with open(path_to_pickle_files + 'macro_model_pickle' , 'rb') as f:
            macro_model = pickle.load(f)

    dash_app = dash.Dash(server=server, routes_pathname_prefix="/dashapp/",)
    fig = go.Figure()

    # Custom HTML layout
    dash_app.index_string = html_layout

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
        yaxis_title='Federal Funds Rate', plot_bgcolor = 'white')

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#ECECEC', zeroline=True, zerolinecolor='lightgrey')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#ECECEC', zeroline=True, zerolinecolor='lightgrey')

    dash_app.layout = html.Div([
                    dcc.Graph(id='plot', figure=fig, style={'display': 'inline-block',
                                                            'height':'75vh',
                                                            'width': '100%'
                                                            }),
                    # slider
                    html.Div([
                        html.Label("Coefficient", style={'margin': '0 0 1% 1%'}),
                        dcc.Slider(id = 'slider',
                                        step = 0.01,
                                        min = -2,
                                        max = 2,
                                        value = 1.65,
                                        tooltip={"placement": "bottom", "always_visible": True},
                                        ),
                        html.P(id='metrics', style={'margin': '0 0 0 1%'})], 
                    style = {'width' : '100%',
                            'fontSize' : '20px',
                            'padding' : '0 2% 0 0',
                            'margin': '0 1% 0 0',
                            'display': 'inline-block',
                            'height': '25vh',
                            'background-color': 'white'
                            },),                   
    ], style={'max-height': '100vh'})

    # Initialize callbacks
    init_callbacks(dash_app, macro_data, training_data, y_train_to_plot, y_perf)
    return dash_app.server

def init_callbacks(app, macro_data, training_data,y_train_to_plot, y_perf):
    """
    Initialize the callbacks used in the app
    """
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
        # 95% confidence interval
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

        # 99% confidence interval
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
                            yaxis_title='Federal Funds Rate', plot_bgcolor = 'white')
        
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#ECECEC', zeroline=True, zerolinecolor='lightgrey')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#ECECEC', zeroline=True, zerolinecolor='lightgrey')

        # calculate R2 and RMSE
        r2 = r2_score(macro_data.y_test, new_y_pred)
        rmse = math.sqrt(mean_squared_error(macro_data.y_test, new_y_pred))

        metrics_report = f"R2: {r2:.3f}, RMSE: {rmse:.3f}"
        return fig ,metrics_report