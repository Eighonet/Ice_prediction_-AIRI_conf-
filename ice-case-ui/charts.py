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

mapbox_token = "sk.eyJ1IjoiZ29yY2hha292diIsImEiOiJjbDV5NWhiYmEwMTUxM2ptYXR4ZHJxZDE5In0.rkVGQw0UehnOJMlXKZpMCA"

def barents_map(model, date):
    print(model, date)
    print('start barents')

    fig = go.Figure()
    #days_list = [i for i in range(int(re.split('-', str(date))[-1]), int(re.split('-', str(date))[-1]) + 1)]
    days_list = [20,21]
    for day in days_list:
        f = open('C:\\Users\\vygorchakov\\Documents\\projects\\Ice_prediction_-AIRI_conf-\\inferences\\models_output\\barents_{}_{}_no_gfs_default.pt.json'.format(date, model))
        day_heatmap = json.load(f)

        f.close()
        fig.add_trace(go.Densitymapbox(lat= day_heatmap['coords']['lats'],
                                         lon=day_heatmap['coords']['lons'],
                                         z=day_heatmap['magn']['2021-06-{}'.format(day + 1)],
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
        currentvalue={"prefix": "Frequency: "},
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
    print('end barents')
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON
