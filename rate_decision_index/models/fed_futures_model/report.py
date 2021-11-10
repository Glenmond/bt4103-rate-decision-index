import pandas as pd
import numpy as np
import os
from calendar import monthrange, month_name
from datetime import datetime,timedelta
import re

import matplotlib as plt
import plotly
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


class Report():
    def __init__(self, result, path):
        self.result = result
        self.path = path

    def generate_prediction_graphs(self, save=False):
        titles = ["Model Results"]  + list(y.strftime("%d %B %Y FOMC Meeting") for y in self.result.index)

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
        table_df = self.result.copy().round(2).reset_index()
        table_df['index'] = pd.to_datetime(table_df['index']).apply(lambda x: x.strftime("%Y-%m-%d"))
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
        df = self.result.copy().round(3)
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

        fig.show()

        today = datetime.now().strftime("%y%m%d")
        if save:
            fig.write_image(f"{self.path}/report/{today}_fff_report.pdf")
        