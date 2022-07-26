import shapely.geometry
import numpy as np
import plotly
import json
import math
import plotly.graph_objects as go
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import re
import sys

from inferences.inference import main_inference


mapbox_token = "sk.eyJ1IjoiZ29yY2hha292diIsImEiOiJjbDV5NWhiYmEwMTUxM2ptYXR4ZHJxZDE5In0.rkVGQw0UehnOJMlXKZpMCA"

def barents_map(model, date):

    #try:
    if date == None or model == None:
        model = 'UNet'
        date = '2021-06-12'
    fig = go.Figure()
    days_list = [i for i in range(int(re.split('-', str(date))[-1]) + 1, int(re.split('-', str(date))[-1]) + 4)]
    day_heatmap = main_inference("barents", str(date), "{}_no_gfs_default.pt".format(model))
    for day in days_list:
        date = '-'.join(re.split('-', str(date))[:-1] + [str(day)])
        magn = [0  if i < 10 else i for i in day_heatmap['magn'][date]]
        magn = [100  if i > 100 else i for i in magn]
        fig.add_trace(go.Densitymapbox(lat= day_heatmap['coords']['lats'],
                                         lon=day_heatmap['coords']['lons'],
                                         z=magn,
                                         radius=10))

    steps = []
    for i in range(len(days_list)):
        step = dict(
            method="update",
            args=[{"visible": [False] * len(days_list)},
                  {"title": "Slider switched to step: " + str(i)}],  # layout attribute
        )
        step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
        steps.append(step)

    sliders = [dict(
        active=0,
        currentvalue={"prefix": "День: "},
        pad={"t": 50},
        steps=steps
    )]

    fig.update_layout(mapbox_style="stamen-terrain", mapbox_center_lon=38.62)
    fig.update_layout(margin={"r":30,"t":0,"l":0,"b":0})
    fig.update_layout(
        sliders=sliders
    )
    fig.update_layout(
        width=1130,
        height=600,
        margin=dict(l=0, r=0, t=0, b=0),
        mapbox=dict(
            accesstoken=mapbox_token,
            bearing=0,
            center=go.layout.mapbox.Center(
                lat=75.75,
                lon=38.62
            ),
            pitch=0,
            zoom=2
        ),
        showlegend=True
    )
    # except:
    #     fig = go.Figure()
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON


def show_metrics():
    fig = go.Figure()
    try:
        fig.add_trace(
                    go.Bar(name="MAE", x = ['UNet', 'ResNet', 'PredRNN', 'ConvLSTM'], y= ["0", "1", "2", "3"]),
        )

        fig.update_xaxes(title_text="<b>Модель<b>")
        fig.update_yaxes(title_text="<b>МАЕ")

        fig.update_layout(height=300, margin={"r": 0, "t": 0, "l": 0, "b": 0},
                                  font=dict(color='rgb(51, 51, 51)', family="arial", size=15),
                                   paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

    except:
        pass

    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON